#ifndef GST_PUSH_HPP
#define GST_PUSH_HPP

#include <memory>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <gst/gst.h>
#include <opencv2/opencv.hpp>

class GstStreamer {
public:
    // 初始化参数结构体
    struct StreamConfig {
        std::string pipeline_desc;  // GStreamer pipeline字符串
        std::string appsrc_name;    // appsrc名称
        int width = 640;            // 视频宽度
        int height = 480;           // 视频高度
        int fps = 30;               // 帧率
        std::string format = "BGR"; // 像素格式
    };

    explicit GstStreamer(const StreamConfig& config)
        : config_(config), is_running_(false) {
        init_gstreamer();
    }

    ~GstStreamer() {
        stop();
    }

    // 启动推流线程
    void start() {
        if (is_running_) return;
        
        is_running_ = true;
        worker_thread_ = std::thread([this](){ 
            this->run_main_loop(); 
        });
    }

    // 停止推流
    void stop() {
        if (!is_running_) return;
        
        is_running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // 推送帧数据（线程安全）
    bool push_frame(const cv::Mat& frame) {
        if (!is_running_) return false;
        
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!frame.empty()) {
            frame_buffer_ = frame.clone();
        }
        return true;
    }

private:
    StreamConfig config_;
    std::atomic<bool> is_running_;
    std::thread worker_thread_;
    cv::Mat frame_buffer_;
    std::mutex frame_mutex_;

    // GStreamer相关成员
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GMainLoop* main_loop_ = nullptr;
    guint bus_watch_id_ = 0;
    guint timeout_id_ = 0;

    void init_gstreamer() {
        // 初始化GStreamer
        if (!gst_is_initialized()) {
            gst_init(nullptr, nullptr);
        }

        // 创建pipeline
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(config_.pipeline_desc.c_str(), &error);
        if (error) {
            throw std::runtime_error("Pipeline创建失败: " + std::string(error->message));
        }

        // 获取appsrc元素
        appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), config_.appsrc_name.c_str());
        if (!appsrc_) {
            gst_object_unref(pipeline_);
            throw std::runtime_error("无法获取appsrc元素");
        }

        // 配置appsrc参数
        GstCaps* caps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, config_.format.c_str(),
            "width", G_TYPE_INT, config_.width,
            "height", G_TYPE_INT, config_.height,
            "framerate", GST_TYPE_FRACTION, config_.fps, 1,
            nullptr);

        g_object_set(appsrc_,
            "caps", caps,
            "block", TRUE,
            "stream-type", 0,
            "format", GST_FORMAT_TIME,
            nullptr);

        gst_caps_unref(caps);

        // 设置总线监听
        GstBus* bus = gst_element_get_bus(pipeline_);
        bus_watch_id_ = gst_bus_add_watch(bus, &GstStreamer::bus_callback, this);
        gst_object_unref(bus);
    }

    void run_main_loop() {
        main_loop_ = g_main_loop_new(nullptr, FALSE);

        // 启动定时推送
        timeout_id_ = g_timeout_add(1000 / config_.fps, 
            [](gpointer user_data) -> gboolean {
                auto self = static_cast<GstStreamer*>(user_data);
                return self->on_push_data();
            }, this);

        // 启动pipeline
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        // 运行主循环
        g_main_loop_run(main_loop_);

        // 清理资源
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        g_source_remove(timeout_id_);
        g_source_remove(bus_watch_id_);
        gst_object_unref(pipeline_);
        g_main_loop_unref(main_loop_);
    }

    gboolean on_push_data() {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (frame_buffer_.empty()) return TRUE;
            frame = frame_buffer_.clone();
        }

        // 确保数据连续性
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }

        // 创建GStreamer缓冲区
        GstBuffer* buffer = gst_buffer_new_wrapped_full(
            GST_MEMORY_FLAG_READONLY,
            frame.data,
            frame.total() * frame.elemSize(),
            0,
            frame.total() * frame.elemSize(),
            nullptr,
            nullptr);

        // 设置时间戳
        static guint64 frame_number = 0;
        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frame_number, GST_SECOND, config_.fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, config_.fps);
        frame_number++;

        // 推送数据
        GstFlowReturn ret;
        g_signal_emit_by_name(appsrc_, "push-buffer", buffer, &ret);
        gst_buffer_unref(buffer);

        if (ret != GST_FLOW_OK) {
            g_printerr("推送缓冲区失败: %d\n", ret);
            return FALSE;
        }

        return TRUE;
    }

    static gboolean bus_callback(GstBus* bus, GstMessage* msg, gpointer data) {
        auto self = static_cast<GstStreamer*>(data);
        
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_EOS:
                g_print("流结束\n");
                g_main_loop_quit(self->main_loop_);
                break;
            case GST_MESSAGE_ERROR: {
                GError* err;
                gchar* debug;
                gst_message_parse_error(msg, &err, &debug);
                g_printerr("错误: %s\n详情: %s\n", err->message, debug);
                g_clear_error(&err);
                g_free(debug);
                g_main_loop_quit(self->main_loop_);
                break;
            }
            case GST_MESSAGE_STATE_CHANGED: {
                GstState old_state, new_state, pending_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                g_print("状态变更: %s -> %s\n",
                    gst_element_state_get_name(old_state),
                    gst_element_state_get_name(new_state));
                break;
            }
            default:
                break;
        }
        return TRUE;
    }
};

#endif // GST_PUSH_HPP