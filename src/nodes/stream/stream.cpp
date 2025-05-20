#include "nodes/stream/stream.hpp"
#include <filesystem>
// 日志库
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"

#include "common/timer.hpp"

namespace node
{

void StreamNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
{
    for (auto &output_queue : output_queues_)
    {
        for (auto &data : batch_datas)
        {
            output_queue.second->push(data);
        }
    }
}

bool StreamNode::open_stream(DecodeType decode_type)
{
    if (decode_type == DecodeType::GPU)
    {
        demuxer_ = FFHDDemuxer::create_ffmpeg_demuxer(stream_url_);
        if (demuxer_ == nullptr)
        {
            PLOGE.printf("StreamNode [%s] Error: GPU demuxer creation failed for %s",
                         name_.c_str(),
                         stream_url_.c_str());
            status_ = StreamStatus::OPEN_FAILED;
            return false;
        }
        fps_ = demuxer_->get_fps();
        PLOGI.printf("StreamNode [%s]: Stream fps %d", name_.c_str(), fps_);

        auto codec_id = demuxer_->get_video_codec();

        decoder_ = FFHDDecoder::create_cuvid_decoder(false,
                                                     FFHDDecoder::ffmpeg2NvCodecId(codec_id),
                                                     -1,
                                                     gpu_id_,
                                                     nullptr,
                                                     nullptr,
                                                     true);

        if (decoder_ == nullptr)
        {
            PLOGE.printf("StreamNode [%s] Error: GPU decoder creation failed for %s (Codec: %d)",
                         name_.c_str(),
                         stream_url_.c_str(),
                         codec_id);
            demuxer_.reset(); // Clean up demuxer if decoder fails
            status_ = StreamStatus::OPEN_FAILED;
            return false;
        }
        PLOGI.printf("StreamNode [%s]: GPU Demuxer and Decoder created successfully.", name_.c_str());
        status_ = StreamStatus::OPENED;
    }
    else if (decode_type_ == DecodeType::CPU)
    {
        printf("stream_url_ : %s\n", stream_url_.c_str());
        cap_ = std::make_shared<cv::VideoCapture>();
        if (!cap_->open(stream_url_))
        {
            PLOGI.printf("StreamNode [%s] Error: CPU cv::VideoCapture failed to open %s",
                         name_.c_str(),
                         stream_url_.c_str());
            cap_.reset(); // Release the failed object
            status_ = StreamStatus::OPEN_FAILED;
            return false;
        }
        if (!cap_->isOpened()) // Double check
        {
            PLOGE.printf("StreamNode [%s] Error: CPU cv::VideoCapture not opened after call for %s",
                         name_.c_str(),
                         stream_url_.c_str());
            cap_.reset();
            status_ = StreamStatus::OPEN_FAILED;
            return false;
        }
        PLOGI.printf("StreamNode [%s]: CPU cv::VideoCapture opened successfully.", name_.c_str());
        status_ = StreamStatus::OPENED;
    }
    else
    {
        status_ = StreamStatus::OPENED;
        return true;
    }
    frame_count_ = 0;
    return true;
}

void StreamNode::close_stream()
{
    PLOGI.printf("StreamNode [%s]: Closing stream...", name_.c_str());
    if (cap_)
    {
        cap_.reset();
    }
    if (decoder_)
    {
        decoder_.reset();
    }
    if (demuxer_)
    {
        demuxer_.reset();
    }
    status_      = StreamStatus::CLOSED;
    frame_count_ = 0;
}

void StreamNode::work()
{
    PLOGI.printf("StreamNode [%s] starting work loop. Decode type: %s",
                 name_.c_str(),
                 (decode_type_ == DecodeType::GPU ? "GPU" : "CPU"));
    while (running_)
    {
        if (status_ != StreamStatus::OPENED)
        {
            PLOGI.printf("StreamNode [%s]: Stream not open (Status: %d). Attempting to open...",
                         name_.c_str(),
                         static_cast<int>(status_));

            if (open_stream(decode_type_))
            {
                PLOGI.printf("StreamNode [%s]: Stream opened successfully.", name_.c_str());
            }
            else
            {
                PLOGI.printf("StreamNode [%s]: Failed to open stream. Retrying in %d ms...",
                             name_.c_str(),
                             retry_delay_ms_);
                status_ = StreamStatus::OPEN_FAILED; // Ensure status reflects failure

                auto wakeUpTime = std::chrono::steady_clock::now() + std::chrono::milliseconds(retry_delay_ms_);
                while (running_ && std::chrono::steady_clock::now() < wakeUpTime)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep in smaller chunks
                }

                if (!running_)
                    break;
            }
        }

        if (status_ == StreamStatus::OPENED)
        {
            PLOGI.printf("StreamNode [%s]: Starting stream processing...", name_.c_str());
            if (decode_type_ == DecodeType::CPU)
            {
                process_stream_cpu();
            }
            else if (decode_type_ == DecodeType::GPU)
            {
                process_stream_gpu();
            }
            else
            {
                process_stream_folder();
                break;
            }

            PLOGI.printf("StreamNode [%s]: Stream processing finished or stopped (Status: %d).",
                         name_.c_str(),
                         static_cast<int>(status_));

            if (status_ == StreamStatus::CLOSED || status_ == StreamStatus::ERROR)
            {
                close_stream();
                PLOGI.printf("StreamNode [%s]: Stream closed or errored. Will attempt reconnection if running.",
                             name_.c_str());
            }
        }
        else
        {
            PLOGD.printf("StreamNode [%s]: Unexpected status %d in work loop.",
                         name_.c_str(),
                         static_cast<int>(status_));
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Avoid tight loop on unexpected state
        }
    }

    PLOGI.printf("StreamNode [%s] work loop finished.", name_.c_str());
    close_stream();
}

void StreamNode::process_stream_folder()
{
    std::vector<std::string> files;
    if (!std::filesystem::exists(stream_url_) && !std::filesystem::is_directory(stream_url_))
    {
        running_.exchange(false);
    }
    else
    {
        for (const auto &entry : std::filesystem::directory_iterator(stream_url_))
        {
            if (entry.is_regular_file())
            {
                files.push_back(entry.path().string());
            }
        }
    }
    for (const auto &file : files)
    {
        cv::Mat frame;
        try
        {
            frame = cv::imread(file);
        }
        catch (const cv::Exception &ex)
        {
            PLOGE.printf("StreamNode [%s] Error: Exception during cv::imread: %s", name_.c_str(), ex.what());
            continue;
        }

        if (frame.empty())
        {
            continue;
        }

        frame_count_++;
        if (frame_count_ % skip_frame_ != 0)
        {
            continue; // Skip frame
        }

        auto frame_data    = std::make_shared<common::FrameData>();
        frame_data->image  = frame.clone();
        frame_data->width  = frame.cols;
        frame_data->height = frame.rows;
        frame_data->timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        frame_data->from        = name_;
        frame_data->fps         = fps_; // Set FPS for folder processing
        frame_data->pipeline_id = owner_pipeline_id_;
        send_single_data_to_output_queues(frame_data);
    }
    status_ == StreamStatus::CLOSED;
    PLOGI.printf("StreamNode [%s]: Exiting FOLDER processing loop (Running: %s, Status: %d).",
                 name_.c_str(),
                 running_ ? "true" : "false",
                 static_cast<int>(status_));
}

void StreamNode::process_stream_cpu()
{
    if (!cap_ || !cap_->isOpened())
    {
        PLOGD.printf("StreamNode [%s] Error: process_stream_cpu called with closed/invalid VideoCapture.",
                     name_.c_str());
        status_ = StreamStatus::ERROR; // Indicate an unexpected state
        return;
    }
    fps_ = cap_->get(cv::CAP_PROP_FPS);

    PLOGI.printf("StreamNode [%s]: Stream fps %d", name_.c_str(), fps_);
    PLOGI.printf("StreamNode [%s]: Processing CPU stream...", name_.c_str());
    while (running_ && status_ == StreamStatus::OPENED)
    {
        cv::Mat frame;
        bool success = false;
        try
        {
            success = cap_->read(frame);
        }
        catch (const cv::Exception &ex)
        {
            PLOGE.printf("StreamNode [%s] Error: Exception during cv::VideoCapture::read(): %s",
                         name_.c_str(),
                         ex.what());
            status_ = StreamStatus::ERROR;
            break;
        }

        if (!success || frame.empty())
        {
            PLOGE.printf("StreamNode [%s]: Cannot read frame (End of stream or error).", name_.c_str());
            status_ = StreamStatus::CLOSED;
            break;
        }

        frame_count_++;
        // if (frame_count_ % skip_frame_ != 0)
        // {
        //     continue; // Skip frame
        // }

        auto frame_data    = std::make_shared<common::FrameData>();
        frame_data->image  = frame.clone();
        frame_data->width  = frame.cols;
        frame_data->height = frame.rows;
        frame_data->timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        frame_data->from        = name_;
        frame_data->fps         = fps_;
        frame_data->pipeline_id = owner_pipeline_id_;

        send_single_data_to_output_queues(frame_data);
    }
    PLOGI.printf("StreamNode [%s]: Exiting CPU processing loop (Running: %s, Status: %d).",
                 name_.c_str(),
                 running_ ? "true" : "false",
                 static_cast<int>(status_));
}

void StreamNode::process_stream_gpu()
{
    if (!demuxer_ || !decoder_)
    {
        PLOGE.printf("StreamNode [%s] Error: process_stream_gpu called with invalid demuxer/decoder.", name_.c_str());
        status_ = StreamStatus::ERROR;
        return;
    }

    PLOGI.printf("StreamNode [%s]: Processing GPU stream...", name_.c_str());
    uint8_t *packet_data = nullptr;
    int packet_size      = 0;
    int64_t pts          = 0;

    // Send extradata once (important for some codecs)
    demuxer_->get_extra_data(&packet_data, &packet_size);
    if (packet_size > 0)
    {
        PLOGI.printf("StreamNode [%s]: Sending %d bytes of extradata to decoder.", name_.c_str(), packet_size);
        decoder_->decode(packet_data, packet_size);
    }
    else
    {
        PLOGI.printf("StreamNode [%s]: No extradata found or needed.", name_.c_str());
    }

    unsigned int frame_index = 0; // Consider if this should be member if state needs preserving across reconnects
    while (running_ && status_ == StreamStatus::OPENED)
    {
        Timer timer(name_); // If using Timer utility
        // Demux next packet
        bool demux_ok = false;
        try
        {
            demux_ok = demuxer_->demux(&packet_data, &packet_size, &pts);
        }
        catch (const std::exception &ex)
        { // Catch potential exceptions from demuxer implementation
            PLOGE.printf("StreamNode [%s] Error: Exception during demuxer_->demux(): %s", name_.c_str(), ex.what());
            status_ = StreamStatus::ERROR;
            break;
        }

        if (!demux_ok || packet_size <= 0 || !running_) // Check running_ again after potentially blocking demux call
        {
            PLOGI.printf("StreamNode [%s]: Demuxing finished or failed (packet_size: %d, running: %s).",
                         name_.c_str(),
                         packet_size,
                         running_ ? "true" : "false");
            status_ = StreamStatus::CLOSED; // Assume normal end or recoverable error
            break;                          // Exit processing loop
        }

        // Decode the packet
        int ndecoded_frame = 0;
        try
        {
            ndecoded_frame = decoder_->decode(packet_data, packet_size, pts);
        }
        catch (const std::exception &ex)
        {
            PLOGE.printf("StreamNode [%s] Error: Exception during decoder_->decode(): %s", name_.c_str(), ex.what());
            status_ = StreamStatus::ERROR;
            break;
        }

        if (ndecoded_frame < 0)
        {
            PLOGE.printf("StreamNode [%s] Error: Decoder returned error (%d).", name_.c_str(), ndecoded_frame);
            status_ = StreamStatus::ERROR; // Treat decoder error as critical
            break;                         // Exit processing loop
        }

        // Process decoded frames
        for (int i = 0; i < ndecoded_frame; ++i)
        {
            // Timer timer("StreamNode GPU Frame"); // If using Timer utility
            uint8_t *frame_data = nullptr;
            int64_t frame_pts   = 0;

            try
            {
                // Pass pointers to get the actual index and PTS for the current frame
                frame_data = decoder_->get_frame(&frame_pts, &frame_index);
            }
            catch (const std::exception &ex)
            {
                PLOGE.printf("StreamNode [%s] Error: Exception during decoder_->get_frame(): %s",
                             name_.c_str(),
                             ex.what());
                status_        = StreamStatus::ERROR;
                ndecoded_frame = 0; // Stop processing frames from this packet
                break;              // Break inner frame loop
            }

            if (!frame_data)
            {
                PLOGE.printf("StreamNode [%s] Error: Decoder returned null frame data for frame %d.", name_.c_str(), i);
                status_        = StreamStatus::ERROR; // Treat null frame data as error
                ndecoded_frame = 0;                   // Stop processing frames from this packet
                break;                                // Break inner frame loop
            }

            // Update overall frame count and check skip logic
            frame_count_++;
            if (frame_count_ % skip_frame_ != 0)
            {
                continue; // Skip this decoded frame
            }

            cv::Mat frame(decoder_->get_height(), decoder_->get_width(), CV_8UC3, frame_data);

            // Create metadata and copy the frame data
            auto frame_meta_data       = std::make_shared<common::FrameData>();
            frame_meta_data->image     = frame.clone(); // CLONE is crucial here!
            frame_meta_data->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count();
            frame_meta_data->from        = name_;
            frame_meta_data->fps         = fps_; // Use demuxer FPS for consistency
            frame_meta_data->width       = decoder_->get_width();
            frame_meta_data->height      = decoder_->get_height();
            frame_meta_data->pipeline_id = owner_pipeline_id_;
            send_single_data_to_output_queues(frame_meta_data);
        }
        if (status_ == StreamStatus::ERROR)
        {
            break;
        }
    };

    PLOGI.printf("StreamNode [%s]: Exiting GPU processing loop (Running: %s, Status: %d, Total frames processed this "
                 "session: %d).",
                 name_.c_str(),
                 running_ ? "true" : "false",
                 static_cast<int>(status_),
                 frame_count_ + 1);
}

} // namespace node