#include "nodes/base.hpp"
#include "common/config.hpp"
#include "common/data.hpp"
#include "common/queue.hpp"
#include "common/timer.hpp"
#include "common/format.hpp"
// 日志库
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"

namespace node
{

// 运行中动态添加输入输出队列可能会导致数据丢失或线程安全问题
// 需要在节点初始化时添加输入输出队列，运行中只需要使用即可
void BaseNode::add_input_queue(const std::string &name, std::shared_ptr<SharedQueue<common::FrameDataPtr>> frame_queue)
{
    std::unique_lock<std::mutex> lock(mutex_);
    frame_queue->set_node_worker_cond(node_worker_cond_);
    input_queues_.insert(std::pair(name, frame_queue));
}

void BaseNode::add_output_queue(const std::string &name, std::shared_ptr<SharedQueue<common::FrameDataPtr>> frame_queue)
{
    std::unique_lock<std::mutex> lock(mutex_);
    output_queues_.insert(std::pair(name, frame_queue));
}

void BaseNode::send_single_data_to_output_queues(const common::FrameDataPtr &data)
{
    for (auto &item : output_queues_)
    {
        item.second->push(data);
    }
}

void BaseNode::send_data_to_output_queues(const std::vector<common::FrameDataPtr> &batch_datas)
{
    for (auto &data : batch_datas)
    {
        if (!data)
        {
            continue;
        }
        send_single_data_to_output_queues(data);
    }
}

void BaseNode::start()
{
    if (!running_.exchange(true))
    {
        worker_thread_ = std::thread(&BaseNode::work, this);
    }
}

void BaseNode::stop()
{
    if (running_.exchange(false))
    {
        // 删除队列中全部元素
        std::for_each(input_queues_.begin(), input_queues_.end(), [&](const auto &item) { item.second->clear(); });
        std::for_each(output_queues_.begin(), output_queues_.end(), [&](const auto &item) { item.second->clear(); });
        node_worker_cond_->notify_all();
        if (worker_thread_.joinable())
        {
            worker_thread_.join();
        }
        PLOGI.printf("Node : [%s] stopped", name_.c_str());
    }
}

void BaseNode::work()
{
    while (running_)
    {
        bool has_data = false;
        for (auto &input_queue : input_queues_)
        {
            std::vector<common::FrameDataPtr> frame_datas;
            input_queue.second->pop_batch(frame_datas, max_pop_batch_size_);
            if (frame_datas.empty())
            {
                continue;
            }
            has_data = true;
            std::string time_name = fmt::str_format("%s %d images", name_.c_str(), frame_datas.size());
            Timer timer(time_name);
            handle_data(frame_datas);
            timer.stop_print();
            send_data_to_output_queues(frame_datas);
        }
        if (!has_data)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            node_worker_cond_->wait_for(lock, std::chrono::milliseconds(100), [this] { return !running_; });
        }
    }
}

} // namespace node