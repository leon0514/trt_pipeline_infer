#include "nodes/track/track.hpp"

#include "common/config.hpp"
#include "common/data.hpp"

namespace node
{

void TrackNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
{
    if (batch_datas.empty())
    {
        return;
    }
    auto track_config_data = std::dynamic_pointer_cast<common::TrackingConfigData>(config_data_);
    for (auto &data : batch_datas)
    {
        std::vector<Object> objects;
        object::DetectionResultArray all_boxes;
        for (const auto &box : data->detection_results)
        {
            all_boxes.push_back(box);
        }
        for (const auto &pose : data->pose_results)
        {
            all_boxes.push_back(pose.box);
        }
        for (const auto &seg : data->segmentation_results)
        {
            all_boxes.push_back(seg.box);
        }
        for (const auto &box : all_boxes)
        {
            // 只处理需要的 label
            if (box.class_name == track_config_data->track_label)
            {
                Object obj;
                obj.rect.x      = box.left;
                obj.rect.y      = box.top;
                obj.rect.width  = box.right - box.left;
                obj.rect.height = box.bottom - box.top;
                obj.label       = 0;
                obj.prob        = box.score;

                if (obj.rect.width > 0 && obj.rect.height > 0 && obj.prob > 0)
                {
                    objects.push_back(obj); // 只添加有效的对象
                }
            }
        }
        std::vector<STrack> output_stracks = tracker_->update(objects);
        for (const auto &track : output_stracks)
        {
            const std::vector<float> &tlwh = track.tlwh;
            data->tracking_results.emplace_back(std::move(object::Box(tlwh[0],
                                                                      tlwh[1],
                                                                      tlwh[0] + tlwh[2],
                                                                      tlwh[1] + tlwh[3],
                                                                      track.score,
                                                                      track.track_id,
                                                                      track_config_data->track_label)),
                                                track.track_id);
        }
    }
}

} // namespace node