#ifndef ROUTER_HPP__
#define ROUTER_HPP__

#include "nodes/base.hpp"

namespace node
{

class RouterNode : public BaseNode
{
  public:
    explicit RouterNode(const std::string &name, common::ConfigDataPtr config_data = nullptr)
        : BaseNode(name, config_data)
    {
        type_ = NodeType::ROUTER;
    }
    virtual ~RouterNode() { stop(); };

    // 路由转发数据
    // 多pipeline共享同一个路由节点，路由节点根据数据的pipeline_id将数据转发到对应的输出队列
    // 路由节点是为了前面共享的推理节点能够接受不同pipeline的数据用来批处理，处理之后再转发到对应的pipeline
    void work() override;

    void handle_data(std::vector<common::FrameDataPtr> &batch_datas) override;
};

} // namespace node

#endif // ROUTER_HPP__