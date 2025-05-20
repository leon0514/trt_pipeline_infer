#ifndef BASE_HPP__
#define BASE_HPP__
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/config.hpp"
#include "common/data.hpp"
#include "common/queue.hpp"

namespace node
{

// 节点类型枚举
// 开始节点 SOURCE
// 中间节点 MIDDLE
// 结束节点 SINK
// 路由节点 ROUTER
// router
// 是一个特殊的节点类型，用于在节点之间进行数据路由分发。用于解决多路输出到对应的pipeline的问题
// 设置该节点的目的是为了推理节点能够接受不同pipeline的数据用来批处理，处理之后再转发到对应的pipeline
enum class NodeType : int
{
    SOURCE = 0,
    MIDDLE = 1,
    SINK   = 2,
    ROUTER = 3
};

// 节点状态
// RUNNING 运行中
// STOP 停止
// ERROR 错误
enum class NodeStatus : int
{
    RUNNING = 1,
    STOP    = 2,
    ERROR   = 3
};

class BaseNode
{

  public:
    BaseNode() = delete;
    explicit BaseNode(const std::string &name, common::ConfigDataPtr config_data)
        : name_(name), config_data_(config_data)
    {
        max_pop_batch_size_ = config_data->max_pop_batch_size_;
    }
    virtual ~BaseNode() = default;

    virtual void work();                                                          // 节点工作函数
    virtual void handle_data(std::vector<common::FrameDataPtr> &batch_datas) = 0; // 节点数据处理函数

    std::string get_name() const { return name_; } // 获取节点名称
    NodeType get_type() const { return type_; }    // 获取节点类型

    NodeStatus get_status() const { return node_status_; }

    void start(); // 启动节点
    void stop();  // 停止节点

    void add_input_queue(const std::string &name,
                         std::shared_ptr<SharedQueue<common::FrameDataPtr>> frame_queue); // 添加输入队列
    void add_output_queue(const std::string &name,
                          std::shared_ptr<SharedQueue<common::FrameDataPtr>> frame_queue); // 添加输出队列
    void send_data_to_output_queues(const std::vector<common::FrameDataPtr> &batch_datas);

    void send_single_data_to_output_queues(const common::FrameDataPtr &batch_datas);

    common::ConfigDataPtr get_config_data() const { return config_data_; }

    void set_config_data(common::ConfigDataPtr config_data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        config_data_ = config_data;
    }

  protected:
    std::string name_; // 节点名称
    NodeType type_;    // 节点类型

    NodeStatus node_status_; // 节点状态

    int max_pop_batch_size_ = 1;

    std::thread worker_thread_;        // 节点工作线程
    std::atomic<bool> running_{false}; // 停止标志
    std::mutex mutex_;                 // 节点锁

    std::shared_ptr<std::condition_variable> node_worker_cond_ = std::make_shared<std::condition_variable>();
    // 节点工作条件变量
    std::unordered_map<std::string, std::shared_ptr<SharedQueue<common::FrameDataPtr>>> input_queues_;  // 输入队列
    std::unordered_map<std::string, std::shared_ptr<SharedQueue<common::FrameDataPtr>>> output_queues_; // 输出队列

    common::ConfigDataPtr config_data_; // 节点配置数据
};

static inline void LinkNode(const std::shared_ptr<BaseNode> &front,
                            const std::shared_ptr<BaseNode> &back,
                            const std::string pipeline_id,
                            int queue_size            = 40,
                            OverflowStrategy strategy = OverflowStrategy::DROP_LATE)
{
    // PLOGI.printf("Link Node %s --> %s", front->get_name().c_str(), back->get_name().c_str());
    auto queue = std::make_shared<SharedQueue<common::FrameDataPtr>>(pipeline_id, queue_size, strategy);
    back->add_input_queue(front->get_name(), queue);
    front->add_output_queue(back->get_name(), queue);
}

} // namespace node

#endif // BASE_HPP__