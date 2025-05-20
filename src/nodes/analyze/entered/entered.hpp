#ifndef ENTERED_HPP__
#define ENTERED_HPP__
#include "nodes/analyze/taskAnalyzer.hpp"
#include "common/polygon.hpp"
#include "common/config.hpp"

class EnteredAnalyzer : public ITaskAnalyzer
{

public:
    EnteredAnalyzer() = default;
    ~EnteredAnalyzer() override = default;

    void analyze(std::vector<common::FrameDataPtr>& batch_datas,
        const std::vector<std::vector<std::tuple<float, float>>>& fences_coords) override
    {
        bool use_geofences = !fences_coords.empty(); // 检查电子围栏是否为空
        std::vector<Polygon> fence_polygons;
        if (use_geofences) 
        {
            fence_polygons.reserve(fences_coords.size());
            for (size_t i = 0; i < fences_coords.size(); ++i) 
            {
                fence_polygons.emplace_back(fences_coords[i], "fence_" + std::to_string(i));
            }
        }

        for (auto& frame_data_ptr : batch_datas) 
        {
            if (!frame_data_ptr) { // 防御性编程，检查空指针
                std::cerr << "Warning: Encountered null FrameDataPtr in batch." << std::endl;
                continue;
            }
            common::FrameData& frame_data = *frame_data_ptr; // 解引用，方便访问
            
            frame_data.fences = fences_coords;
            for (const auto& detection : frame_data.pose_results) 
            {
                if (detection.box.class_name != "person") 
                {
                    continue; // 只处理 "person" 目标
                }

                if (use_geofences) 
                {
                    Polygon box_polygon({
                        {detection.box.left, detection.box.top},
                        {detection.box.right, detection.box.top},
                        {detection.box.right, detection.box.bottom},
                        {detection.box.left, detection.box.bottom}
                    }, "detection_box");

                    double box_area = box_polygon.getArea();
                    if (box_area < 1e-6) { // 避免除以零或非常小的面积导致IOU不稳定
                        // 如果检测框面积过小，根据业务逻辑决定是否直接跳过或认为不在围栏内
                        // std::cout << "Warning: Detection box area is near zero." << std::endl;
                        continue;
                    }

                    bool inside_any_fence = false;
                    for (const auto& fence_poly : fence_polygons) 
                    {
                        // 注意：这里的 getIntersectionArea 需要确保 Polygon 类已经正确实现
                        double intersection_area = box_polygon.getIntersectionArea(fence_poly);
                        double iou = intersection_area / box_area; // 使用检测框自己的面积作为分母来计算IOU
                                                                    // 或者使用联合面积: intersection_area / (box_area + fence_poly.getArea() - intersection_area)
                                                                    // 取决于你对IOU的定义，此处用检测框面积更符合“目标在围栏内的重叠比例”
                        if (iou > 0.5) { // IOU 阈值
                            inside_any_fence = true;
                            break; // 只要在一个围栏内满足条件即可
                        }
                    }

                    if (inside_any_fence) 
                    {
                        auto result_box = detection.box;
                        result_box.class_name = "entered";
                        frame_data.results.push_back(result_box);
                    }
                } 
                else 
                {
                    auto result_box = detection.box;
                    result_box.class_name = "entered";
                    frame_data.results.push_back(result_box);
                }
            }
        }
    }
};

#endif // PERSONCOUNT_HPP__