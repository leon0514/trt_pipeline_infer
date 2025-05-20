#ifndef DATA_HPP__
#define DATA_HPP__
#include <chrono>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/object.hpp"

namespace common
{

class FrameData
{
  public:
    explicit FrameData() {}
    ~FrameData() = default;

  public:
    std::string pipeline_id; // 所属的pipeline ID
    int64_t timestamp;       // 获取到该图片的时间 单位：毫秒
    cv::Mat image;           // 图像数据
    cv::Mat osd_image;       // 图像数据
    int width;               // 图像宽度
    int height;              // 图像高度
    std::string from;        // 数据来源
    int fps;                 // 视频fps

  public:
    object::DetectionResultArray detection_results;        // 检测结果
    object::DetectionObbResultArray detection_obb_results; // 旋转框检测结果
    object::PoseResultArray pose_results;                  // 姿态估计结果
    object::SegmentationResultArray segmentation_results;  // 分割结果
    object::TrackingResultArray tracking_results;          // 跟踪结果

    object::FenceArray fences;   // 图片中设置的电子围栏
    object::ResultArray results; // 最终结果
};

// 智能指针类型定义

using FrameDataPtr = std::shared_ptr<FrameData>;

} // namespace common

#endif // DATA_HPP__