#include "nodes/osd/osd.hpp"
#include "common/config.hpp"
#include "common/format.hpp"
#include "nodes/base.hpp"
#include "nodes/osd/position.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <ctime>

// 日志库
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"

namespace node
{

const std::vector<std::pair<int, int>> coco_pairs = {{0, 1},
                                                     {0, 2},
                                                     {0, 11},
                                                     {0, 12},
                                                     {1, 3},
                                                     {2, 4},
                                                     {5, 6},
                                                     {5, 7},
                                                     {7, 9},
                                                     {6, 8},
                                                     {8, 10},
                                                     {11, 12},
                                                     {5, 11},
                                                     {6, 12},
                                                     {11, 13},
                                                     {13, 15},
                                                     {12, 14},
                                                     {14, 16}};

static std::tuple<int, int, int> getFontSize(const std::string &text)
{
    int baseline      = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);

    return std::make_tuple(textSize.width, textSize.height, baseline);
}

static void
overlay_mask(cv::Mat &image, const cv::Mat &smallMask, int roiX, int roiY, const cv::Scalar &color, double alpha)
{
    if (image.empty() || smallMask.empty() || image.type() != CV_8UC3 || smallMask.type() != CV_8UC1)
    {
        return;
    }
    alpha = std::max(0.0, std::min(1.0, alpha));

    cv::Rect roiRect(roiX, roiY, smallMask.cols, smallMask.rows);

    cv::Rect imageRect(0, 0, image.cols, image.rows);
    cv::Rect intersectionRect = roiRect & imageRect; // 使用 & 操作符计算交集

    if (intersectionRect.width <= 0 || intersectionRect.height <= 0)
    {
        return;
    }

    cv::Mat originalROI = image(intersectionRect); // ROI 指向 image 的数据

    int maskStartX = intersectionRect.x - roiX;
    int maskStartY = intersectionRect.y - roiY;
    cv::Rect maskIntersectionRect(maskStartX, maskStartY, intersectionRect.width, intersectionRect.height);
    cv::Mat smallMaskROI = smallMask(maskIntersectionRect);

    cv::Mat colorPatchROI(intersectionRect.size(), image.type(), color);

    cv::Mat tempColoredROI = originalROI.clone(); // 需要一个临时区域进行覆盖
    colorPatchROI.copyTo(tempColoredROI, smallMaskROI);

    cv::addWeighted(originalROI, 1.0 - alpha, tempColoredROI, alpha, 0.0, originalROI);
}

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i)
    {
    case 0:
        r = v, g = t, b = p;
        break;
    case 1:
        r = q, g = v, b = p;
        break;
    case 2:
        r = p, g = v, b = t;
        break;
    case 3:
        r = p, g = q, b = v;
        break;
    case 4:
        r = t, g = p, b = v;
        break;
    case 5:
        r = v, g = p, b = q;
        break;
    default:
        r = 1, g = 1, b = 1;
        break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static void osd_box(cv::Mat &image, const object::Box &box, PositionManager<float> &pm)
{
    int id     = box.class_id;
    auto color = random_color(id);
    cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));
    cv::rectangle(image, box.getRect(), bgr_color, 2);
    int x, y;
    std::string text = fmt::str_format("%s %.2f", box.class_name.c_str(), box.score);
    std::tie(x, y)   = pm.selectOptimalPosition(std::make_tuple(box.left, box.top, box.right, box.bottom),
                                              image.cols,
                                              image.rows,
                                              text);
    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 2);
}

static void osd_obbox(cv::Mat &image, const object::OBBox &box)
{
    int id     = box.class_id;
    auto color = random_color(id);
    cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));

    // Draw the oriented bounding box
    std::vector<cv::Point> vertices(4);
    for (int i = 0; i < 4; ++i)
    {
        float x, y;
        std::tie(x, y) = box.point(i);
        vertices[i]    = cv::Point(static_cast<int>(x), static_cast<int>(y));
    }
    cv::polylines(image, std::vector<std::vector<cv::Point>>{vertices}, true, bgr_color, 2);

    // Draw the label and score directly above the box
    std::string text = fmt::str_format("%s %.2f", box.class_name.c_str(), box.score);
    float left, top;
    std::tie(left, top) = box.left_top();
    // Find center point
    cv::Point center = vertices[0];
    for (int i = 1; i < 4; ++i)
    {
        center += vertices[i];
    }
    center.x /= 4;
    center.y /= 4;

    // Draw text at center position
    cv::putText(image, text, cv::Point(center.x - 20, center.y), cv::FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 2);
}

// FenceArray std::vector<std::vector<std::tuple<float, float>>>
void OsdNode::osd_fences(cv::Mat &image, const object::FenceArray &fences)
{
    cv::Scalar fence_color(0, 255, 0); // Green color for fences

    for (size_t i = 0; i < fences.size(); i++)
    {
        const auto &fence = fences[i];
        if (fence.size() < 2) continue; // Need at least 2 points to draw a line

        std::vector<cv::Point> fence_points;
        for (const auto &point : fence)
        {
            float x, y;
            std::tie(x, y) = point;
            fence_points.emplace_back(static_cast<int>(x), static_cast<int>(y));
        }

        float x, y;
        std::tie(x, y) = fence[0];
        fence_points.emplace_back(static_cast<int>(x), static_cast<int>(y));

        // Draw lines between consecutive points
        for (size_t j = 1; j < fence_points.size(); j++)
        {
            cv::line(image, fence_points[j - 1], fence_points[j], fence_color, 2);
        }
    }
}

void OsdNode::osd_detection(cv::Mat &image, const object::DetectionResultArray &detection_results)
{
    PositionManager<float> pm(getFontSize);
    for (const auto &box : detection_results)
    {
        osd_box(image, box, pm);
    }
}

void OsdNode::osd_obb(cv::Mat &image, const object::DetectionObbResultArray &detection_obb_results)
{
    for (const auto &obbox : detection_obb_results)
    {
        osd_obbox(image, obbox);
    }
}

void OsdNode::osd_pose(cv::Mat &image, const object::PoseResultArray &pose_results)
{
    PositionManager<float> pm(getFontSize);
    for (const auto &pose : pose_results)
    {
        int id     = pose.box.class_id;
        auto color = random_color(id);
        cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));
        osd_box(image, pose.box, pm);
        for (const auto &pair : coco_pairs)
        {
            auto first  = pose.keypoints[pair.first].to_cv_point();
            auto second = pose.keypoints[pair.second].to_cv_point();
            cv::line(image, first, second, bgr_color, 2);
        }
        for (int i = 0; i < pose.keypoints.size(); ++i)
        {
            cv::circle(image, pose.keypoints[i].to_cv_point(), 3, bgr_color, -1);
        }
    }
}

void OsdNode::osd_segmentation(cv::Mat &image, const object::SegmentationResultArray &segmentation_results)
{
    PositionManager<float> pm(getFontSize);
    for (const auto &segment : segmentation_results)
    {
        osd_box(image, segment.box, pm);
        int id     = segment.box.class_id;
        auto color = random_color(id);
        cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));
        if (segment.seg == nullptr)
        {
            continue;
        }
        cv::Mat mask(segment.seg->height, segment.seg->width, CV_8UC1, segment.seg->data);
        overlay_mask(image, mask, segment.box.left, segment.box.top, bgr_color, 0.6);
    }
}

void OsdNode::osd_tracking(cv::Mat &image, const object::TrackingResultArray &tracking_results)
{
    auto now = std::chrono::steady_clock::now();
    for (const auto &track : tracking_results)
    {
        int id     = track.track_id;
        auto color = random_color(id);
        cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));

        cv::Point center((track.box.left + track.box.right) / 2, (track.box.top + track.box.bottom) / 2);
        std::string track_text = fmt::str_format("ID: %d", track.box.class_id);
        cv::putText(image, track_text, center, cv::FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2);
        track_history_[track.box.class_id].push_back(center);

        if (track_history_[track.box.class_id].size() > 50)
        {
            track_history_[track.box.class_id].erase(track_history_[track.box.class_id].begin());
        }

        const auto &history = track_history_[track.box.class_id];
        for (size_t i = 1; i < history.size(); ++i)
        {
            cv::line(image, history[i - 1], history[i], bgr_color, 2);
        }
        last_track_update_time_[track.box.class_id] = now;
    }
    for (auto it = track_history_.begin(); it != track_history_.end();)
    {
        if (last_track_update_time_.find(it->first) != last_track_update_time_.end() &&
            std::chrono::duration_cast<std::chrono::seconds>(now - last_track_update_time_[it->first]).count() > 5)
        {
            last_track_update_time_.erase(it->first);
            it = track_history_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void OsdNode::osd_time(cv::Mat &image, int64_t& timestamp, int x, int y)
{
    if (image.empty()) {
        return;
    }

   long long total_milliseconds = timestamp;
   long long milliseconds_part = total_milliseconds % 1000;
   std::time_t seconds_since_epoch = static_cast<std::time_t>(total_milliseconds / 1000);

   std::tm* timeinfo_utc;
   timeinfo_utc = std::gmtime(&seconds_since_epoch);

   if (!timeinfo_utc) {
       std::string error_msg = "Invalid Timestamp";
       cv::putText(image, error_msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 1, cv::LINE_AA);
       return;
   }

   std::ostringstream time_ss;
   time_ss << (timeinfo_utc->tm_year + 1900) << "-" // tm_year 是自1900年以来的年数
           << std::setw(2) << std::setfill('0') << (timeinfo_utc->tm_mon + 1) << "-" // tm_mon 是0-11的月份
           << std::setw(2) << std::setfill('0') << timeinfo_utc->tm_mday << " "
           << std::setw(2) << std::setfill('0') << timeinfo_utc->tm_hour << ":"
           << std::setw(2) << std::setfill('0') << timeinfo_utc->tm_min << ":"
           << std::setw(2) << std::setfill('0') << timeinfo_utc->tm_sec << "."
           << std::setw(3) << std::setfill('0') << milliseconds_part;
   std::string time_str = time_ss.str();

    int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体类型
    double fontScale = 0.7;                  // 字体大小
    cv::Scalar color(0, 255, 0);             // 字体颜色 (BGR格式，这里是绿色)
    int thickness = 1;                       // 字体粗细
    int lineType = cv::LINE_AA;              // 线条类型 (抗锯齿)

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(time_str, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // 左上角位置 (x, y)，y坐标是基线的位置
    cv::Point textOrg(x, y + textSize.height);

    // 6. 在图像上绘制文本
    cv::putText(image,          // 目标图像
                time_str,       // 要绘制的文本
                textOrg,        // 文本框的左下角坐标 (cv::putText的org是左下角)
                fontFace,       // 字体
                fontScale,      // 字体缩放因子
                color,          // 文本颜色
                thickness,      // 文本线条的粗细
                lineType);      // 线条类型
}

void OsdNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
{
    auto config_data          = std::dynamic_pointer_cast<common::OsdConfigData>(config_data_);
    bool show_final_result    = config_data->show_final_result;
    bool show_original_result = config_data->show_original_result;

    for (auto &frame_data : batch_datas)
    {
        frame_data->osd_image = frame_data->image.clone();
        if (config_data_ == nullptr)
        {
            PLOGE.printf("DrawNode : [%s] config data is null, show nothing", name_.c_str());
            return;
        }

        cv::Mat image    = frame_data->osd_image;
        int image_width  = image.cols;
        int image_height = image.rows;

        // 画原始识别结果 检测框、关键点、分割、跟踪id、跟踪轨迹
        if (show_original_result)
        {
            osd_detection(image, frame_data->detection_results);
            osd_pose(image, frame_data->pose_results);
            osd_obb(image, frame_data->detection_obb_results);
            osd_segmentation(image, frame_data->segmentation_results);
            osd_tracking(image, frame_data->tracking_results);
        }

        if (show_final_result)
        {
            osd_detection(image, frame_data->results);
            osd_fences(image, frame_data->fences);
        }
        // osd_time(image, frame_data->timestamp, 10, 10);
        // int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     std::chrono::system_clock::now().time_since_epoch())
        //     .count();
        // osd_time(image, now, 10, 30);
    }
}

} // namespace node
