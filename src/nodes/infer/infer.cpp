#include "nodes/infer/infer.hpp"

namespace node
{

void InferNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
{
    // printf("batch_datas size: %d\n", batch_datas.size());
    std::vector<cv::Mat> iput_images;
    iput_images.reserve(batch_datas.size());
    for (auto &data : batch_datas)
    {
        iput_images.push_back(data->image);
    }
    InferResult det_res = model_->forwards(iput_images);

    std::visit(
        [&batch_datas](auto &&result)
        {
            int batch_size = batch_datas.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::DetectionResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    auto &target_vector    = batch_datas[i]->detection_results;
                    auto &source_vector    = result[i]; // non-const 左值引用
                    size_t elements_to_add = source_vector.size();

                    if (elements_to_add > 0)
                    {
                        target_vector.reserve(target_vector.size() + elements_to_add);
                        for (auto &element : source_vector)
                        {
                            target_vector.push_back(std::move(element));
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<object::PoseResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    auto &target_vector    = batch_datas[i]->pose_results;
                    auto &source_vector    = result[i]; // non-const 左值引用
                    size_t elements_to_add = source_vector.size();

                    if (elements_to_add > 0)
                    {
                        target_vector.reserve(target_vector.size() + elements_to_add);
                        for (auto &element : source_vector)
                        {
                            target_vector.push_back(std::move(element));
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<object::DetectionObbResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    auto &target_vector    = batch_datas[i]->detection_obb_results;
                    auto &source_vector    = result[i]; // non-const 左值引用
                    size_t elements_to_add = source_vector.size();

                    if (elements_to_add > 0)
                    {
                        target_vector.reserve(target_vector.size() + elements_to_add);
                        for (auto &element : source_vector)
                        {
                            target_vector.push_back(std::move(element));
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<object::SegmentationResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    auto &target_vector    = batch_datas[i]->segmentation_results;
                    auto &source_vector    = result[i]; // non-const 左值引用
                    size_t elements_to_add = source_vector.size();

                    if (elements_to_add > 0)
                    {
                        target_vector.reserve(target_vector.size() + elements_to_add);
                        for (auto &element : source_vector)
                        {
                            target_vector.push_back(std::move(element));
                        }
                    }
                }
            }
        },
        det_res);
}
} // namespace node