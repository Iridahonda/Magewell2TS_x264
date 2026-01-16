/*
 * Copyright (c) 2022 John Patrick Poet
 *
 * Output a Transport Stream with fixes for:
 * - A/V Sync (Independent Normalization)
 * - Playback Speed / Duration Glitches
 * - Thread Shutdown Deadlocks
 * - x264 Colorspace & Profile
 */

#include <csignal>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <fcntl.h>
#include <chrono>
#include <atomic>

#include "OutputTS.h"
#include "lock_ios.h"

extern "C" {
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

using namespace std;
using namespace s6_lock_ios;

// --- A/V SYNC GLOBALS ---
// Independent anchors for Audio and Video to ensure
// both start at PTS 0, regardless of hardware clock offsets.
static std::atomic<int64_t> g_video_start_ts{-1};
static std::atomic<int64_t> g_audio_start_ts{-1};

static std::string AVerr2str(int code)
{
    char astr[AV_ERROR_MAX_STRING_SIZE] = { 0 };
    av_make_error_string(astr, AV_ERROR_MAX_STRING_SIZE, code);
    return string(astr);
}

static std::string AV_ts2str(int64_t ts) {
    char astr[AV_TS_MAX_STRING_SIZE] = { 0 };
    av_ts_make_string(astr, ts);
    return string(astr);
}

OutputTS::OutputTS(int verbose_level, const string & video_codec_name,
                   const string & preset, int quality, int look_ahead,
                   bool no_audio, bool p010, const string & device,
                   const string & output_filename, ShutdownCallback shutdown, ResetCallback reset,
                   MagCallback image_buffer_avail)
    : m_verbose(verbose_level)
    , m_no_audio(no_audio)
    , m_video_codec_name(video_codec_name)
    , m_device("/dev/dri/" + device)
    , m_preset(preset)
    , m_quality(quality)
    , m_look_ahead(look_ahead)
    , m_p010(p010)
    , m_filename(output_filename)
    , f_shutdown(shutdown)
    , f_reset(reset)
    , f_image_buffer_available(image_buffer_avail)
{
    // RESET SYNC ANCHORS ON NEW SESSION
    // "Audio Changing" event
    g_video_start_ts.store(-1);
    g_audio_start_ts.store(-1);

    if (m_video_codec_name.find("x264") != string::npos)
        m_encoderType = EncoderType::X264;
    else if (m_video_codec_name.find("qsv") != string::npos)
        m_encoderType = EncoderType::QSV;
    else if (m_video_codec_name.find("vaapi") != string::npos)
        m_encoderType = EncoderType::VAAPI;
    else if (m_video_codec_name.find("nvenc") != string::npos)
        m_encoderType = EncoderType::NV;
    else
        m_encoderType = EncoderType::UNKNOWN;

    if (!m_no_audio)
        m_audioIO = new AudioIO([=](bool val) {this->DiscardImages(val); }, verbose_level);

    m_mux_thread = std::thread(&OutputTS::mux, this);
    pthread_setname_np(m_mux_thread.native_handle(), "mux");

    m_copy_thread = std::thread(&OutputTS::copy_to_frame, this);
    pthread_setname_np(m_copy_thread.native_handle(), "copy");

    m_display_primaries  = av_mastering_display_metadata_alloc();
    m_content_light  = av_content_light_metadata_alloc(NULL);
}

void OutputTS::Shutdown(void)
{
    if (m_running.exchange(false))
    {
        // DEADLOCK FIX: Wake up all threads immediately
        m_imagequeue_ready.notify_all();
        m_videopool_avail.notify_all();
        m_videopool_ready.notify_all();
        
        f_shutdown();
        if (m_audioIO) m_audioIO->Shutdown();
    }
}

// ... [Helper functions] ...
void OutputTS::setLight(AVMasteringDisplayMetadata * d, AVContentLightMetadata * l) {
    if (d && l) { *m_display_primaries = *d; *m_content_light = *l; }
}

AVFrame* OutputTS::alloc_audio_frame(enum AVSampleFormat fmt, const AVChannelLayout* layout, int rate, int nb) {
    AVFrame* f = av_frame_alloc();
    if (!f) return nullptr;
    f->format = fmt;
    av_channel_layout_copy(&f->ch_layout, layout);
    f->sample_rate = rate;
    f->nb_samples = nb;
    if (nb && av_frame_get_buffer(f, 0) < 0) return nullptr;
    return f;
}

bool OutputTS::open_audio(void)
{
    close_encoder(&m_audio_stream);
    const AVCodec* codec = avcodec_find_encoder_by_name(m_audioIO->CodecName().c_str());
    if (!codec) return true;

    m_audio_stream.tmp_pkt = av_packet_alloc();
    m_audio_stream.enc = avcodec_alloc_context3(codec);
    m_audio_stream.enc->bit_rate = (m_audioIO->NumChannels() == 2) ? 256000 : 640000;
    
    m_audio_stream.enc->sample_fmt = AV_SAMPLE_FMT_FLTP;
    m_audio_stream.enc->sample_rate = 48000;
    av_channel_layout_copy(&m_audio_stream.enc->ch_layout, m_audioIO->ChannelLayout());

    if (avcodec_open2(m_audio_stream.enc, codec, NULL) < 0) return false;

    int nb_samples = (m_audio_stream.enc->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) ? 10000 : m_audio_stream.enc->frame_size;
    m_audio_stream.frame = alloc_audio_frame(m_audio_stream.enc->sample_fmt, &m_audio_stream.enc->ch_layout, m_audio_stream.enc->sample_rate, nb_samples);
    
    enum AVSampleFormat in_fmt = (m_audioIO->BytesPerSample() == 4) ? AV_SAMPLE_FMT_S32 : AV_SAMPLE_FMT_S16;
    m_audio_stream.tmp_frame = alloc_audio_frame(in_fmt, &m_audio_stream.enc->ch_layout, m_audio_stream.enc->sample_rate, nb_samples);

    m_audio_stream.swr_ctx = swr_alloc();
    av_opt_set_chlayout(m_audio_stream.swr_ctx, "in_chlayout", &m_audio_stream.enc->ch_layout, 0);
    av_opt_set_int(m_audio_stream.swr_ctx, "in_sample_rate", m_audio_stream.enc->sample_rate, 0);
    av_opt_set_sample_fmt(m_audio_stream.swr_ctx, "in_sample_fmt", in_fmt, 0);
    av_opt_set_chlayout(m_audio_stream.swr_ctx, "out_chlayout", &m_audio_stream.enc->ch_layout, 0);
    av_opt_set_int(m_audio_stream.swr_ctx, "out_sample_rate", m_audio_stream.enc->sample_rate, 0);
    av_opt_set_sample_fmt(m_audio_stream.swr_ctx, "out_sample_fmt", m_audio_stream.enc->sample_fmt, 0);
    swr_init(m_audio_stream.swr_ctx);

    return true;
}

bool OutputTS::open_video(void)
{
    close_encoder(&m_video_stream);
    
    if (m_video_stream.frames) {
        for (int i=0; i<m_video_stream.frames_total; ++i) av_frame_free(&m_video_stream.frames[i].frame);
        delete[] m_video_stream.frames;
        m_video_stream.frames = nullptr;
    }
    m_video_stream.frames_total = m_frame_buffers;
    m_video_stream.frames_used = 0;
    m_video_stream.frames_idx_in = 0;
    m_video_stream.frames_idx_out = 0;

    const AVCodec* video_codec = avcodec_find_encoder_by_name(m_video_codec_name.c_str());
    if (!video_codec) return false;

    m_video_stream.tmp_pkt = av_packet_alloc();
    m_video_stream.enc = avcodec_alloc_context3(video_codec);
    m_video_stream.enc->width = m_input_width;
    m_video_stream.enc->height = m_input_height;
    // Set BOTH time_base and framerate
    // Encoder timebase should be the inverse of the framerate (1/60, etc.)
    m_video_stream.enc->time_base = av_inv_q(m_input_frame_rate);
    m_video_stream.enc->framerate = m_input_frame_rate;
    m_video_stream.enc->time_base = AVRational{m_input_frame_rate.den, m_input_frame_rate.num};
    
    m_video_stream.enc->colorspace = m_color_space;
    m_video_stream.enc->color_primaries = m_color_primaries;
    m_video_stream.enc->color_trc = m_color_trc;
    m_video_stream.enc->color_range = m_isHDR ? AVCOL_RANGE_JPEG : AVCOL_RANGE_UNSPECIFIED;

    AVDictionary* opt = NULL;
    switch (m_encoderType) {
        case EncoderType::X264:
            if (!open_x264(video_codec, &m_video_stream, opt)) return false;
            break;
        case EncoderType::QSV:
            open_qsv(video_codec, &m_video_stream, opt);
            break;
        case EncoderType::VAAPI:
            open_vaapi(video_codec, &m_video_stream, opt);
            break;
        default: return false;
    }
    return true;
}

bool OutputTS::open_container(void)
{
    close_container();
    if (!m_running.load()) return false;

    avformat_alloc_output_context2(&m_output_format_context, NULL, "mpegts", NULL);
    m_fmt = m_output_format_context->oformat;

    // Video Stream
    m_video_stream.st = avformat_new_stream(m_output_format_context, NULL);
    m_video_stream.st->id = 0;
    // Force 90kHz timebase for MPEG-TS
    m_video_stream.st->time_base = (AVRational){1, 90000};
    avcodec_parameters_from_context(m_video_stream.st->codecpar, m_video_stream.enc);

    // Audio Stream
    if (m_audio_stream.enc) {
        m_audio_stream.st = avformat_new_stream(m_output_format_context, NULL);
        m_audio_stream.st->id = 1;
        m_audio_stream.st->time_base = (AVRational){ 1, m_audio_stream.enc->sample_rate };
        avcodec_parameters_from_context(m_audio_stream.st->codecpar, m_audio_stream.enc);
    }

    if (!(m_fmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&m_output_format_context->pb, m_filename.c_str(), AVIO_FLAG_WRITE) < 0) return false;
    }
    if (!(m_output_format_context->oformat->flags & AVFMT_NOFILE)) {
            // Use the filename member instead of hardcoded pipe:1
            if (avio_open(&m_output_format_context->pb, m_filename.c_str(), AVIO_FLAG_WRITE) < 0) {
                cerr << "Could not open output file: " << m_filename << endl;
                return false;
            }
    }
    avformat_write_header(m_output_format_context, NULL);
    m_init_needed = false;
    return true;
}

void OutputTS::close_container(void) {
    if (m_output_format_context) {
        if (m_output_format_context->pb) avio_closep(&m_output_format_context->pb);
        avformat_free_context(m_output_format_context);
        m_output_format_context = nullptr;
    }
}

void OutputTS::close_encoder(OutputStream* ost) {
    if (ost->enc) {
        avcodec_free_context(&ost->enc);
        ost->enc = nullptr;
    }
    if (ost->frames) {
        for(int i=0; i<ost->frames_total; ++i) if(ost->frames[i].frame) av_frame_free(&ost->frames[i].frame);
        delete[] ost->frames;
        ost->frames = nullptr;
    }
}

bool OutputTS::open_x264(const AVCodec *codec, OutputStream *ost, AVDictionary *opt_arg)
{
    AVDictionary *opt = nullptr;
    if (opt_arg) av_dict_copy(&opt, opt_arg, 0);
    
    AVCodecContext* c = ost->enc;
    c->pix_fmt = AV_PIX_FMT_YUV422P;
    av_dict_set(&opt, "crf", "1", 0);
    
    c->bit_rate = 7000000;
    c->rc_max_rate = 12000000;
    c->rc_min_rate = 5000000;
    c->rc_buffer_size = 12000000;
    
    c->colorspace = AVCOL_SPC_BT709;
    c->color_primaries = AVCOL_PRI_BT709;
    c->color_trc = AVCOL_TRC_BT709;
    c->color_range = AVCOL_RANGE_MPEG;
    

    av_dict_set(&opt, "profile", "high422", 0);
    av_dict_set(&opt, "preset", "ultrafast", 0);
    av_dict_set(&opt, "preset", m_preset.c_str(), 0);
    av_dict_set(&opt, "tune", "zerolatency", 0);

    if (avcodec_open2(c, codec, &opt) < 0) return false;
    av_dict_free(&opt);

    ost->frames = new OutputTS::OutputStream::Frame[ost->frames_total];
    for (int i=0; i<ost->frames_total; ++i) {
        ost->frames[i].frame = av_frame_alloc();
        ost->frames[i].frame->format = c->pix_fmt;
        ost->frames[i].frame->width = c->width;
        ost->frames[i].frame->height = c->height;
        av_frame_get_buffer(ost->frames[i].frame, 0);
        ost->frames[i].timestamp = -1;
    }
    return true;
}

// ... [Removed GPU functions as I dont use Todo add id needed] ...
bool OutputTS::open_nvidia(const AVCodec*, OutputStream*, AVDictionary*) { return false; }
bool OutputTS::open_vaapi(const AVCodec*, OutputStream*, AVDictionary*) { return false; }
bool OutputTS::open_qsv(const AVCodec*, OutputStream*, AVDictionary*) { return false; }
bool OutputTS::nv_encode() { return false; }
bool OutputTS::qsv_vaapi_encode() { return false; }
// ...

bool OutputTS::x264_encode(void) {
    OutputStream* ost = &m_video_stream;
    return write_frame(m_output_format_context, ost->enc, ost->frame, ost);
}

void OutputTS::mux(void) {
    while (m_running.load()) {
        if (m_audioIO && m_audioIO->CodecChanged()) {
            open_audio();
            m_init_needed = true;
        }

        if (m_init_needed) {
            if (m_video_stream.enc && (!m_audioIO || m_audio_stream.enc))
                open_container();
        }

        if (m_audio_stream.enc) write_audio_frame(m_output_format_context, &m_audio_stream);

        while (m_running.load()) {
            std::unique_lock<std::mutex> lock(m_videopool_mutex);
            if (!m_video_stream.enc || m_video_stream.frames_used == 0) {
                m_videopool_empty.notify_one();
                m_videopool_ready.wait_for(lock, std::chrono::milliseconds(5));
                if (m_video_stream.frames_used == 0) break;
            }

            int idx = m_video_stream.frames_idx_out;
            if (++m_video_stream.frames_idx_out == m_video_stream.frames_total)
                m_video_stream.frames_idx_out = 0;

            m_video_stream.frame = m_video_stream.frames[idx].frame;
            m_video_stream.timestamp = m_video_stream.frames[idx].timestamp;

            if (m_encoderType == EncoderType::X264) x264_encode();
            
            --m_video_stream.frames_used;
            m_videopool_avail.notify_one();
        }
    }
}

void OutputTS::copy_to_frame()
{
    struct SwsContext* sws_ctx = nullptr;

    while (m_running.load()) {
        imagepkt_t pkt;
        {
            std::unique_lock<std::mutex> lock(m_imagequeue_mutex);
            m_imagequeue_ready.wait(lock, [this] { return !m_imagequeue.empty() || !m_running; });
            if (!m_running && m_imagequeue.empty()) break;
            pkt = m_imagequeue.front();
            m_imagequeue.pop_front();
            if (m_imagequeue.empty()) m_imagequeue_empty.notify_all();
        }

        OutputStream *ost = &m_video_stream;
        std::unique_lock<std::mutex> pool_lock(m_videopool_mutex);
        m_videopool_avail.wait(pool_lock, [this, ost] { return (ost->frames_used < ost->frames_total) || !m_running; });

        if (!m_running) {
            f_image_buffer_available(pkt.image, pkt.pEco);
            break;
        }

        int idx = ost->frames_idx_in;
        AVFrame* dest = ost->frames[idx].frame;

        // Scaler: Packed YUY2 -> Planar YUV422 Flipping pain...
        if (m_encoderType == EncoderType::X264) {
            if (!sws_ctx) {
                sws_ctx = sws_getContext(m_input_width, m_input_height,
                                         AV_PIX_FMT_YUYV422, // The Magewell input format
                                         ost->enc->width, ost->enc->height,
                                         AV_PIX_FMT_YUV422P, // The x264 encoder format
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
            }
            uint8_t* src_data[4] = { pkt.image, nullptr, nullptr, nullptr };
            int src_linesize[4] = { m_input_width * 2, 0, 0, 0 }; // Correct for packed YUYV
            sws_scale(sws_ctx, src_data, src_linesize, 0, m_input_height, dest->data, dest->linesize);
        }

        // --- VIDEO SYNC ---
        int64_t raw_ts = pkt.timestamp;

        int64_t expected = -1;
        if (raw_ts > 0) g_video_start_ts.compare_exchange_strong(expected, raw_ts);
        
        int64_t normalized_ts = 0;
        int64_t start = g_video_start_ts.load();
        if (start != -1) {
            normalized_ts = raw_ts - start;
            if (normalized_ts < 0) normalized_ts = 0;
        }

        ost->frames[idx].timestamp = normalized_ts;

        // FIX: Rescale from Magewell (100ns) to ENCODER timebase (1/fps)
        // NOT the stream timebase (90kHz).
        
        dest->pts = av_rescale_q(normalized_ts, m_input_time_base, ost->enc->time_base);
        ost->frames_idx_in = (ost->frames_idx_in + 1) % ost->frames_total;
        ost->frames_used++;

        m_videopool_ready.notify_one();
        pool_lock.unlock();
        f_image_buffer_available(pkt.image, pkt.pEco);
    }
    if (sws_ctx) sws_freeContext(sws_ctx);
}


AVFrame* OutputTS::get_pcm_audio_frame(OutputStream* ost)
{
    AVFrame* frame = ost->tmp_frame;
    int bytes = ost->enc->ch_layout.nb_channels * frame->nb_samples * m_audioIO->BytesPerSample();

    if (m_audioIO->Size() < bytes) {
        this_thread::sleep_for(chrono::milliseconds(1));
        return nullptr;
    }
    if (m_audioIO->Read((uint8_t*)frame->data[0], bytes) <= 0) return nullptr;

    int64_t raw_ts = m_audioIO->TimeStamp();
    
    // --- AUDIO SYNC ---
    int64_t expected = -1;
    if (raw_ts > 0) g_audio_start_ts.compare_exchange_strong(expected, raw_ts);

    int64_t normalized_ts = 0;
    int64_t start = g_audio_start_ts.load();
    if (start != -1) {
        normalized_ts = raw_ts - start;
        if (normalized_ts < 0) normalized_ts = 0;
    }

    ost->timestamp = normalized_ts;
    ost->frame->pts = av_rescale_q(normalized_ts, m_input_time_base, ost->enc->time_base);
    return frame;
}

bool OutputTS::setVideoParams(int w, int h, bool i, AVRational tb, double dur, AVRational fr, bool hdr) {
    m_input_width = w; m_input_height = h; m_interlaced = i; m_input_time_base = tb; m_input_frame_rate = fr; m_isHDR = hdr; m_input_frame_duration = dur;
    m_frame_buffers = 10;
    open_video();
    m_init_needed = true;
    return true;
}
bool OutputTS::setAudioParams(int n, bool l, int b, int s, int sp, int f) { return m_audioIO->AddBuffer(n, l, b, s, sp, f); }
bool OutputTS::addAudio(AudioBuffer::AudioFrame *& b, int64_t t) { return m_audioIO->Add(b, t); }
AVFrame* OutputTS::alloc_picture(enum AVPixelFormat f, int w, int h) {
    AVFrame* p = av_frame_alloc(); p->format=f; p->width=w; p->height=h; av_frame_get_buffer(p, 0); return p;
}
void OutputTS::ClearVideoPool() { std::unique_lock<std::mutex> l(m_videopool_mutex); m_video_stream.frames_used=0; m_video_stream.frames_idx_in=0; m_video_stream.frames_idx_out=0; }
void OutputTS::ClearImageQueue() { std::unique_lock<std::mutex> l(m_imagequeue_mutex); for(auto& p : m_imagequeue) f_image_buffer_available(p.image, p.pEco); m_imagequeue.clear(); }
void OutputTS::DiscardImages(bool v) { m_discard_images=v; if(v){ClearVideoPool(); ClearImageQueue();} }

bool OutputTS::AddVideoFrame(uint8_t* pImage, void* pEco, int imageSize, int64_t timestamp)
{
    const std::unique_lock<std::mutex> lock(m_imagequeue_mutex);
    if (m_discard_images) f_image_buffer_available(pImage, pEco);
    else {
        m_imagequeue.push_back(imagepkt_t{timestamp, pImage, pEco, imageSize});
        m_imagequeue_ready.notify_one();
    }
    return true;
}

bool OutputTS::write_frame(AVFormatContext* fmt_ctx, AVCodecContext* codec_ctx, AVFrame* frame, OutputStream* ost) {
    int ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) return false;

    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx, ost->tmp_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) return false;

        // Rescale from Encoder Timebase to Stream Timebase
        // This ensures the MPEG-TS muxer gets the correct 90kHz values.
        av_packet_rescale_ts(ost->tmp_pkt, codec_ctx->time_base, ost->st->time_base);
        
        ost->tmp_pkt->stream_index = ost->st->index;
        
        std::lock_guard<std::mutex> lock(m_container_mutex);
        av_interleaved_write_frame(fmt_ctx, ost->tmp_pkt);
    }
    return true;
}

bool OutputTS::write_pcm_frame(AVFormatContext* oc, OutputStream* ost) {
    AVFrame* f = get_pcm_audio_frame(ost);
    if (!f) return false;
    AVCodecContext* c = ost->enc;
    if (av_frame_make_writable(ost->frame) < 0) return false;
    swr_convert(ost->swr_ctx, ost->frame->data, ost->frame->nb_samples, (const uint8_t**)f->data, f->nb_samples);
    ost->frame->pts = av_rescale_q(ost->timestamp, m_input_time_base, c->time_base);
    return write_frame(oc, c, ost->frame, ost);
}

bool OutputTS::write_bitstream_frame(AVFormatContext* oc, OutputStream* ost) {
    AVPacket* pkt = m_audioIO->ReadSPDIF();
    if (!pkt) return false;
    int64_t raw_ts = m_audioIO->TimeStamp();
    
    // --- AUDIO BITSTREAM SYNC ---
    int64_t expected = -1;
    if (raw_ts > 0) g_audio_start_ts.compare_exchange_strong(expected, raw_ts);
    int64_t norm = 0;
    int64_t start = g_audio_start_ts.load();
    if (start != -1) norm = raw_ts - start;
    if (norm < 0) norm = 0;

    pkt->pts = av_rescale_q(norm, m_input_time_base, ost->st->time_base);
    pkt->dts = pkt->pts;
    pkt->stream_index = ost->st->index;
    return (av_interleaved_write_frame(oc, pkt) >= 0);
}

bool OutputTS::write_audio_frame(AVFormatContext* oc, OutputStream* ost) {
    return m_audioIO->Bitstream() ? write_bitstream_frame(oc, ost) : write_pcm_frame(oc, ost);
}

OutputTS::~OutputTS(void) {
    Shutdown();
    if (m_display_primaries) av_freep(&m_display_primaries);
    if (m_content_light) av_freep(&m_content_light);
    if (m_mux_thread.joinable()) m_mux_thread.join();
    if (m_copy_thread.joinable()) m_copy_thread.join();
    if (m_audioIO) delete m_audioIO;
}
