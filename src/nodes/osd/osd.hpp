#ifndef OSD_HPP__
#define OSD_HPP__
#include "nodes/base.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

namespace node
{

class OsdNode : public BaseNode
{
  public:
    OsdNode() = delete;
    OsdNode(const std::string &name, common::ConfigDataPtr config_data) : BaseNode(name, config_data) {}

    virtual ~OsdNode() { stop(); }

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

  private:
    void osd_detection(cv::Mat &image, const object::DetectionResultArray &detection_results);

    void osd_pose(cv::Mat &image, const object::PoseResultArray &pose_results);

    void osd_obb(cv::Mat &image, const object::DetectionObbResultArray &detection_obb_results);

    void osd_segmentation(cv::Mat &image, const object::SegmentationResultArray &segmentation_results);

    void osd_tracking(cv::Mat &image, const object::TrackingResultArray &tracking_results);

    void osd_fences(cv::Mat &image, const object::FenceArray &fences);

  private:
    std::unordered_map<int, std::vector<cv::Point>> track_history_;
    std::unordered_map<int, std::chrono::steady_clock::time_point> last_track_update_time_;
};

} // namespace node

#endif // DRAWNODE_HPP__