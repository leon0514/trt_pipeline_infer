#ifndef INFERNODE_HPP__
#define INFERNODE_HPP__

#include "common/config.hpp"
#include "nodes/base.hpp"
#include "trt/infer.hpp"

#include <fstream>
#include <iostream>
#include <vector>

namespace node
{

class InferNode : public BaseNode
{
  public:
    InferNode() = delete;
    // RAII
    InferNode(const std::string &name, common::ConfigDataPtr config_data) : BaseNode(name, config_data)
    {
        auto infer_config_data = std::dynamic_pointer_cast<common::InferConfigData>(config_data);
        if (!infer_config_data)
        {
            throw std::bad_cast();
        }
        model_path_             = infer_config_data->model_path;
        model_type_             = ModelTypeConverter::from_string(infer_config_data->model_type);
        gpu_id_                 = infer_config_data->gpu_id;
        confidence_threshold_   = infer_config_data->conf_threshold;
        nms_threshold_          = infer_config_data->nms_threshold;
        max_batch_size_         = infer_config_data->max_batch_size;
        auto_slice_             = infer_config_data->auto_slice;
        slice_width_            = infer_config_data->slice_width;
        slice_height_           = infer_config_data->slice_height;
        slice_horizontal_ratio_ = infer_config_data->slice_horizontal_ratio;
        slice_vertical_ratio_   = infer_config_data->slice_vertical_ratio;

        std::vector<std::string> names;

        if (!infer_config_data->names_path.empty())
        {
            std::ifstream ifs(infer_config_data->names_path);
            if (!ifs.is_open())
            {
                throw std::runtime_error("Failed to open names file: " + infer_config_data->names_path);
            }
            std::string line;
            while (std::getline(ifs, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                names.push_back(line);
            }
            if (ifs.bad())
            {
                throw std::runtime_error("Error reading names file: " + infer_config_data->names_path);
            }
        }
        else
        {
            throw std::runtime_error("Names path is empty.");
        }

        model_ = load(model_path_,
                      model_type_,
                      names,
                      gpu_id_,
                      confidence_threshold_,
                      nms_threshold_,
                      max_batch_size_,
                      auto_slice_,
                      slice_width_,
                      slice_height_,
                      slice_horizontal_ratio_,
                      slice_vertical_ratio_);
    }
    virtual ~InferNode() { stop(); };

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

  private:
    std::shared_ptr<InferBase> model_ = nullptr;
    std::string model_path_;
    ModelType model_type_;
    int gpu_id_;

    bool auto_slice_ = false; // 是否自动切分
    int slice_width_;
    int slice_height_;
    float slice_horizontal_ratio_;
    float slice_vertical_ratio_;

    float confidence_threshold_;
    float nms_threshold_;

    int max_batch_size_ = 1; // 批处理大小
};

} // namespace node

#endif