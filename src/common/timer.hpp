#ifndef TIMER_HPP__
#define TIMER_HPP__

#include <memory>
#include <string>
#include <chrono>   // 用于时间测量
#include <iostream> // 用于打印输出
#include <string>   // 用于计时器名称
#include <iomanip>  // 用于格式化输出 (std::fixed, std::setprecision)


class Timer 
{
public:
    Timer(const std::string& name = "Timer")
        : m_name(name),
            m_startTimePoint(std::chrono::high_resolution_clock::now()),
            m_stopped(false) // 初始化为未停止状态
    {

    }

    ~Timer() {
        if (!m_stopped) {
            stop_print();
        }
    }

    void stop_print() {
        if (m_stopped) { // 如果已经停止并打印过，则直接返回
            return;
        }

        auto endTimePoint = std::chrono::high_resolution_clock::now();

        // 计算持续时间
        auto duration = endTimePoint - m_startTimePoint;

        // 将持续时间转换为更易读的单位（例如：微秒、毫秒、秒）
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double milliseconds = microseconds / 1000.0;
        double seconds = milliseconds / 1000.0;

        printf("%-*s Elapsed time: %-*lld us | %-*.3f ms | %-*.6f s\n",
            18, m_name.c_str(),
            10, static_cast<long long>(microseconds),
            10, milliseconds,
            10, seconds);

        m_stopped = true; // 标记为已停止并打印
    }
    
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(Timer&&) = delete;

private:
    std::string m_name;                                               // 计时器名称
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimePoint; // 开始时间点
    bool m_stopped;                                                   // 标记是否已手动停止并打印
};
    

#endif // TIMER_HPP__