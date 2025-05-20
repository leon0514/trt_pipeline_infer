#ifndef TRACK_HPP__
#define TRACK_HPP__
#include "BYTETracker.h"
#include "common/config.hpp"
#include "nodes/base.hpp"

namespace node
{

class TrackNode : public BaseNode
{
  public:
    explicit TrackNode(const std::string &name, common::ConfigDataPtr config_data) : BaseNode(name, config_data)
    {
        auto track_config_data = std::dynamic_pointer_cast<common::TrackingConfigData>(config_data_);
        type_                  = NodeType::MIDDLE;
        int frame_rate         = track_config_data->frame_rate;                           // 帧率
        int track_buffer       = track_config_data->track_buffer;                         // 跟踪缓冲区大小
        tracker_               = std::make_shared<BYTETracker>(frame_rate, track_buffer); // 创建跟踪器对象
    }
    virtual ~TrackNode() { stop(); };

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

  private:
    std::shared_ptr<BYTETracker> tracker_ = nullptr; // 跟踪器对象
};

} // namespace node

#endif // TRACK_HPP__