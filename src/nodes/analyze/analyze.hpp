#ifndef ANALYZE_HPP__
#define ANALYZE_HPP__
#include "nodes/base.hpp"
#include "nodes/analyze/taskAnalyzer.hpp"
#include "nodes/analyze/entered/entered.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

namespace node
{

class AnalyzeNode : public BaseNode
{
  public:
  AnalyzeNode() = delete;
    AnalyzeNode(const std::string &name, common::AnalyzeConfigDataPtr config_data) : BaseNode(name, config_data) 
    {
      m_analyze_map_ = {
          {"entered", std::make_shared<EnteredAnalyzer>()},
      };
    }

    virtual ~AnalyzeNode() { stop(); }

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;

  private:
    std::unordered_map<std::string, std::shared_ptr<ITaskAnalyzer>> m_analyze_map_;
};

} // namespace node

#endif // ANALYZE_HPP__