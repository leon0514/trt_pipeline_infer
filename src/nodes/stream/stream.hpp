#ifndef STREAM_HPP__
#define STREAM_HPP__
#include "common/config.hpp"
#include "nodes/base.hpp"
#include "nodes/stream/decodeStream.hpp"
#include "opencv2/opencv.hpp"

namespace node
{

enum class DecodeType : int
{
    CPU    = 0,
    GPU    = 1,
    FOLDER = 2
};

enum class StreamStatus
{
    OPENED      = 0,
    CLOSED      = 1,
    OPEN_FAILED = 2,
    ERROR       = 3
};

inline DecodeType stringToDecodeType(const std::string &str)
{
    if (str == "CPU")
        return DecodeType::CPU;
    else if (str == "GPU")
        return DecodeType::GPU;
    else if (str == "FOLDER")
        return DecodeType::FOLDER;
    else
        throw std::invalid_argument("Invalid DecodeType string: " + str);
}

inline std::string decodeTypeToString(DecodeType type)
{
    switch (type)
    {
    case DecodeType::CPU:
        return "CPU";
    case DecodeType::GPU:
        return "GPU";
    case DecodeType::FOLDER:
        return "FOLDER";
    default:
        return "UNKNOWN";
    }
}

inline StreamStatus stringToStreamStatus(const std::string &str)
{
    if (str == "OPENED")
        return StreamStatus::OPENED;
    else if (str == "CLOSED")
        return StreamStatus::CLOSED;
    else if (str == "OPEN_FAILED")
        return StreamStatus::OPEN_FAILED;
    else if (str == "ERROR")
        return StreamStatus::ERROR;
    else
        throw std::invalid_argument("Invalid StreamStatus string: " + str);
}

inline std::string streamStatusToString(StreamStatus status)
{
    switch (status)
    {
    case StreamStatus::OPENED:
        return "OPENED";
    case StreamStatus::CLOSED:
        return "CLOSED";
    case StreamStatus::OPEN_FAILED:
        return "OPEN_FAILED";
    case StreamStatus::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

class StreamNode : public BaseNode
{

  public:
    StreamNode() = delete;
    StreamNode(const std::string &name, common::ConfigDataPtr config_data) : BaseNode(name, config_data)
    {
        type_                   = NodeType::SOURCE;
        auto stream_config_data = std::dynamic_pointer_cast<common::StreamConfigData>(config_data_);
        stream_url_             = stream_config_data->stream_url;
        skip_frame_             = stream_config_data->skip_frame;
        decode_type_            = stringToDecodeType(stream_config_data->decode_type);
        owner_pipeline_id_      = stream_config_data->owner_pipeline_id;
        frame_count_            = 1;
        open_stream(decode_type_);
    }

    virtual ~StreamNode()
    {
        stop();         // Ensure stop is called (should set running_ to false)
        close_stream(); // Clean up resources
    };

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

  public:
    StreamStatus get_status() const { return status_; }

    void work() override;

  private:
    bool open_stream(DecodeType decode_type);
    void close_stream();
    void process_stream_cpu();
    void process_stream_gpu();
    void process_stream_folder();

  private:
    std::string owner_pipeline_id_;
    std::string stream_url_;

    int gpu_id_               = 0;
    int skip_frame_           = 1;
    int retry_delay_ms_       = 500;
    unsigned int frame_count_ = 1;
    int fps_                  = 20;

    std::shared_ptr<cv::VideoCapture> cap_               = nullptr;
    std::shared_ptr<FFHDDemuxer::FFmpegDemuxer> demuxer_ = nullptr;
    std::shared_ptr<FFHDDecoder::CUVIDDecoder> decoder_  = nullptr;

    DecodeType decode_type_ = DecodeType::GPU;
    StreamStatus status_    = StreamStatus::CLOSED;
};

} // end namespace node

#endif