#ifndef RECORD_HPP__
#define RECORD_HPP__

#include "common/config.hpp"
#include "nodes/base.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

// 日志库
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"

namespace node
{

class RecordNode : public BaseNode
{
  public:
    RecordNode() = delete;
    RecordNode(const std::string &name, common::ConfigDataPtr config_data) : BaseNode(name, config_data)
    {
        auto record_config_data = std::dynamic_pointer_cast<common::RecordConfigData>(config_data_);
        gst_pipeline_           = record_config_data->build_gst_pipeline_string();
        if (gst_pipeline_.empty())
        {
            PLOGE.printf("RecordNode : [%s] gst_pipeline_ is empty.", name_.c_str());
            return;
        }
    }

    virtual ~RecordNode() { stop(); }

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

  private:
    int fps_ = 25;

    std::string gst_pipeline_;
    std::shared_ptr<cv::VideoWriter> writer_ = nullptr;
};


// class RecordNode : public BaseNode
// {
//   public:
//     RecordNode() = delete;
//     RecordNode(const std::string &name, common::ConfigDataPtr config_data) : BaseNode(name, config_data)
//     {
//         auto record_config_data = std::dynamic_pointer_cast<common::RecordConfigData>(config_data_);
//         config_.pipeline_desc = record_config_data->build_gst_pipeline_string();
//         config_.appsrc_name = record_config_data->pipeline_elements[0].properties["name"];
//         if (config_.pipeline_desc.empty())
//         {
//             PLOGE.printf("RecordNode : [%s] gst_pipeline_ is empty.", name_.c_str());
//             return;
//         }
//     }

//     virtual ~RecordNode() { stop(); }

//     void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

//   private:
//     int fps_ = 25;
//     GstStreamer::StreamConfig config_;
//     std::string gst_pipeline_;
//     std::shared_ptr<GstStreamer> writer_ = nullptr;
// };


} // namespace node

#endif