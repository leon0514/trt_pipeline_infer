#include "nodes/analyze/analyze.hpp"
#include "common/config.hpp"
#include "common/format.hpp"
#include "nodes/base.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>

// 日志库
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"

namespace node
{

void AnalyzeNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas) 
{
    auto config_data          = std::dynamic_pointer_cast<common::AnalyzeConfigData>(config_data_);

    std::vector<std::vector<std::tuple<float, float>>> fences = config_data->fences;
    std::string task_name = config_data->task_name;

    if (m_analyze_map_.count(task_name))
    {
        m_analyze_map_[task_name]->analyze(batch_datas, fences);
    }
    else
    {
        LOG_ERROR << "Task name " << task_name << " not found in analyze map.";
    }

}

} // namespace node
