#ifndef TASKANALYZER_HPP__
#define TASKANALYZER_HPP__

class ITaskAnalyzer
{

public:
    virtual ~ITaskAnalyzer() = default;
    virtual void analyze(std::vector<common::FrameDataPtr>& batch_datas, const std::vector<std::vector<std::tuple<float, float>>>& fences_coords) = 0;
};

#endif // TASKANALYZER_HPP__