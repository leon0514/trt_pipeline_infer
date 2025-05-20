#ifndef DECODESTREAM_HPP__
#define DECODESTREAM_HPP__

#include <algorithm>

#include <memory>
#include <unordered_map>

#include "opencv2/opencv.hpp"

#include "nodes/stream/ffhdd/cuda_tools.hpp"
#include "nodes/stream/ffhdd/cuvid_decoder.hpp"
#include "nodes/stream/ffhdd/ffmpeg_demuxer.hpp"

class FFmpegDemuxer
{
  public:
    FFmpegDemuxer(std::string uri, bool auto_reboot = false)
    {
        instance_ = FFHDDemuxer::create_ffmpeg_demuxer(uri, auto_reboot);
    }

    bool valid() { return instance_ != nullptr; }

    FFHDDemuxer::IAVCodecID get_video_codec() { return instance_->get_video_codec(); }
    virtual FFHDDemuxer::IAVPixelFormat get_chroma_format() { return instance_->get_chroma_format(); }
    virtual int get_width() { return instance_->get_width(); }
    virtual int get_height() { return instance_->get_height(); }
    virtual int get_bit_depth() { return instance_->get_bit_depth(); }
    virtual int get_fps() { return instance_->get_fps(); }
    virtual int get_total_frames() { return instance_->get_total_frames(); }

    virtual bool isreboot() { return instance_->isreboot(); }
    virtual void reset_reboot_flag() { instance_->reset_reboot_flag(); }

    virtual bool reopen() { return instance_->reopen(); }

  private:
    int64_t time_pts_ = 0;
    std::shared_ptr<FFHDDemuxer::FFmpegDemuxer> instance_;
}; // FFmpegDemuxer

class CUVIDDecoder
{
  public:
    CUVIDDecoder(bool bUseDeviceFrame,
                 FFHDDemuxer::IAVCodecID eCodec,
                 int max_cache,
                 int gpu_id,
                 int cl,
                 int ct,
                 int cr,
                 int cb,
                 int rw,
                 int rh,
                 bool output_bgr)
    {

        FFHDDecoder::IcudaVideoCodec codec = FFHDDecoder::ffmpeg2NvCodecId(eCodec);
        FFHDDecoder::CropRect crop{0, 0, 0, 0};
        FFHDDecoder::ResizeDim resize{0, 0};
        if (cr - cl > 0 && cb - ct > 0)
        {
            crop.l = cl;
            crop.t = ct;
            crop.r = cr;
            crop.b = cb;
        }

        if (rw > 0 && rh > 0)
        {
            resize.w = rw;
            resize.h = rh;
        }

        output_bgr_ = output_bgr;
        instance_ =
            FFHDDecoder::create_cuvid_decoder(bUseDeviceFrame, codec, max_cache, gpu_id, &crop, &resize, output_bgr);
    }

    bool valid() { return instance_ != nullptr; }

    int get_frame_bytes() { return instance_->get_frame_bytes(); }
    int get_width() { return instance_->get_width(); }
    int get_height() { return instance_->get_height(); }
    unsigned int get_frame_index() { return instance_->get_frame_index(); }
    unsigned int get_num_decoded_frame() { return instance_->get_num_decoded_frame(); }

    int decode(uint64_t pData, int nSize, int64_t nTimestamp = 0)
    {
        const uint8_t *ptr = (const uint8_t *)pData;
        return instance_->decode(ptr, nSize, nTimestamp);
    }
    int64_t get_stream() { return (uint64_t)instance_->get_stream(); }

  private:
    std::shared_ptr<FFHDDecoder::CUVIDDecoder> instance_;
    bool output_bgr_ = false;
};

#endif // STREAM_HPP__