#include "nodes/record/record.hpp"

namespace node
{

void RecordNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
{
    if (writer_ == nullptr)
    {
        fps_       = batch_datas[0]->fps;
        int width  = batch_datas[0]->width;
        int height = batch_datas[0]->height;
        PLOGI.printf("RecordNode : [%s] Initializing VideoWriter with pipeline: %s, fps: %d, size: %dx%d",
                     name_.c_str(),
                     gst_pipeline_.c_str(),
                     fps_,
                     width,
                     height);
        if (batch_datas[0]->width <= 0 || batch_datas[0]->height <= 0 || batch_datas[0]->fps <= 0)
        {
            PLOGE.printf("RecordNode : [%s] received frame with invalid dimensions/fps (w:%d, h:%d, fps:%d). Cannot "
                         "initialize writer.",
                         name_.c_str(),
                         batch_datas[0]->width,
                         batch_datas[0]->height,
                         batch_datas[0]->fps);
            return; // 或者调用 stop()
        }

        writer_ = std::make_shared<cv::VideoWriter>();
        // 尝试打开
        writer_->open(gst_pipeline_, cv::CAP_GSTREAMER, 0, fps_, cv::Size(width, height), true); // 假设总是彩色

        // **立即检查** 是否打开成功
        if (!writer_->isOpened())
        {
            PLOGE.printf("RecordNode : [%s] FAILED to open VideoWriter with pipeline: %s",
                         name_.c_str(),
                         gst_pipeline_.c_str());
            writer_ = nullptr;
            return; // 返回，不处理这个批次的数据
        }
        else
        {
            PLOGI.printf("RecordNode : [%s] VideoWriter opened successfully.", name_.c_str());
        }
    }

    if (writer_ == nullptr || !writer_->isOpened())
    {
        PLOGW.printf("RecordNode : [%s] Writer is not initialized or not open, skipping frame writing.", name_.c_str());
        return;
    }

    // 3. 遍历并写入帧数据
    for (const auto &frame_data : batch_datas)
    {
        // **重要：确保 frame_data->image 是有效的 cv::Mat**
        if (frame_data && !frame_data->osd_image.empty())
        {
            writer_->write(frame_data->osd_image);
        }
        else
        {
            PLOGW.printf("RecordNode : [%s] Skipping null or empty frame in the batch.", name_.c_str());
        }
    }
}


// void RecordNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
// {
//     if (writer_ == nullptr)
//     {
//         fps_       = batch_datas[0]->fps;
//         int width  = batch_datas[0]->width;
//         int height = batch_datas[0]->height;
//         config_.width = width;
//         config_.height = height;
//         config_.fps = fps_;

//         PLOGI.printf("RecordNode : [%s] Initializing VideoWriter with pipeline: %s, fps: %d, size: %dx%d",
//                      name_.c_str(),
//                      gst_pipeline_.c_str(),
//                      fps_,
//                      width,
//                      height);
//         if (batch_datas[0]->width <= 0 || batch_datas[0]->height <= 0 || batch_datas[0]->fps <= 0)
//         {
//             PLOGE.printf("RecordNode : [%s] received frame with invalid dimensions/fps (w:%d, h:%d, fps:%d). Cannot "
//                          "initialize writer.",
//                          name_.c_str(),
//                          batch_datas[0]->width,
//                          batch_datas[0]->height,
//                          batch_datas[0]->fps);
//             return; // 或者调用 stop()
//         }

//         writer_ = std::make_shared<GstStreamer>(config_);
//         writer_->start();
//     }

//     // 3. 遍历并写入帧数据
//     for (const auto &frame_data : batch_datas)
//     {
//         // **重要：确保 frame_data->image 是有效的 cv::Mat**
//         if (frame_data && !frame_data->osd_image.empty())
//         {
//             writer_->push_frame(frame_data->osd_image);
//         }
//         else
//         {
//             writer_->stop();
//             writer_ = nullptr;
//             return;
//             PLOGW.printf("RecordNode : [%s] Skipping null or empty frame in the batch.", name_.c_str());
//         }
//     }
// }

} // namespace node