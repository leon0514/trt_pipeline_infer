#ifndef QUEUE_HPP__
#define QUEUE_HPP__

#include <condition_variable>
#include <limits> // 用于 std::numeric_limits
#include <mutex>
#include <queue>
#include <stdexcept> // 用于 std::invalid_argument

/**
 * @brief 定义队列满时的处理策略
 */
enum class OverflowStrategy : int
{
    BLOCK      = 0, // 阻塞等待
    DROP_EARLY = 1, // 丢弃早期数据
    DROP_LATE  = 2, // 丢弃晚期数据
    DROP_ALL   = 3  // 丢弃所有数据
};

template <typename T> class SharedQueue
{

  public:
    explicit SharedQueue(const std::string &owner_pipeline_id,
                         size_t max_size           = 25,
                         OverflowStrategy strategy = OverflowStrategy::DROP_LATE)
        : owner_pipeline_id_(owner_pipeline_id), max_size_(max_size), overflow_strategy_(strategy)
    {
    }

    ~SharedQueue() = default;

    std::string get_pipeline_id() const { return owner_pipeline_id_; }

    void set_pipeline_id(const std::string &pipeline_id) { owner_pipeline_id_ = pipeline_id; }

    void set_node_worker_cond(std::shared_ptr<std::condition_variable> node_worker_cond)
    {
        node_worker_cond_ = node_worker_cond;
    }

    void push(const T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_)
        {
            switch (overflow_strategy_)
            {
            case OverflowStrategy::BLOCK:
                not_full_cond_.wait(lock, [this] { return queue_.size() < max_size_; });
                break;
            case OverflowStrategy::DROP_EARLY:
                queue_.pop();
                break;
            case OverflowStrategy::DROP_LATE:
                return; // 不做任何操作，直接返回
            case OverflowStrategy::DROP_ALL:
                while (!queue_.empty())
                {
                    queue_.pop();
                }
                break;
            }
        }
        queue_.push(item);
        node_worker_cond_->notify_one(); // 有数据进入到队列了，通知工作线程
        not_empty_cond_.notify_one();    // 通知消费者线程
    }

    bool empty()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_cond_.wait(lock, [this] { return !queue_.empty(); });
        item = queue_.front();
        queue_.pop();
        not_full_cond_.notify_one(); // 通知生产者线程
    }

    void pop_batch(std::vector<T> &items, size_t max_batch_size)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (items.size() < max_batch_size && !queue_.empty())
        {
            items.push_back(queue_.front());
            queue_.pop();
        }
        not_full_cond_.notify_one();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty_queue;
        std::swap(queue_, empty_queue); // O(1) 时间清空
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t capacity() const { return max_size_; }

    OverflowStrategy get_overflow_strategy() const { return overflow_strategy_; }

  private:
    std::string owner_pipeline_id_; // 所属的pipeline ID 用于路由节点转发数据
    std::queue<T> queue_;
    size_t max_size_;
    OverflowStrategy overflow_strategy_;
    std::mutex mutex_;

    std::shared_ptr<std::condition_variable> node_worker_cond_;
    std::condition_variable not_empty_cond_;
    std::condition_variable not_full_cond_;
};

#endif // QUEUE_HPP__