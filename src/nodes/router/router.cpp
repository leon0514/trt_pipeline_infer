#include "nodes/router/router.hpp"

namespace node
{

void RouterNode::work()
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
            handle_data(frame_datas);
            // timer.stop_print();
            // send_data_to_output_queues(frame_datas);
        }
        if (!has_data)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            node_worker_cond_->wait_for(lock, std::chrono::milliseconds(100), [this] { return !running_; });
        }
    }
}

void RouterNode::handle_data(std::vector<common::FrameDataPtr> &batch_datas)
{
    for (auto &output_queue : output_queues_)
    {
        for (auto &data : batch_datas)
        {
            if (output_queue.second->get_pipeline_id() == data->pipeline_id)
            {
                // printf("fps : %d, wh : %dx%d\n", data->fps, data->width, data->height);
                output_queue.second->push(data);
            }
        }
    }
}

} // namespace node