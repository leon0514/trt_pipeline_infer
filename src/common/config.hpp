#ifndef CONFIG_HPP__
#define CONFIG_HPP__

#include <chrono>
#include <memory>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <tuple>

namespace common
{

class BaseData
{
  public:
    explicit BaseData(int max_pop_batch_size = 1)
        : create_time_(std::chrono::system_clock::now()), max_pop_batch_size_(max_pop_batch_size)
    {
    }
    virtual ~BaseData() = default;

  public:
    int max_pop_batch_size_;

  protected:
    std::chrono::system_clock::time_point create_time_;
};

class StreamConfigData : public BaseData
{
  public:
    explicit StreamConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~StreamConfigData() override = default;

  public:
    // 所属pipeline id. stream node
    // 属于源头节点，在创造数据的时候需要给数据标明属于的pipeline
    std::string owner_pipeline_id;
    std::string stream_name; // 流名称
    std::string stream_url;  // 流路径
    std::string decode_type; // 解码类型 CPU GPU FOLDER
    int skip_frame = 1;      // 跳过的帧数
    int gpu_id     = -1;     // GPU ID
};

class InferConfigData : public BaseData
{
  public:
    explicit InferConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~InferConfigData() override = default;

  public:
    std::string model_path;        // 模型路径
    std::string model_type;        // 模型类型
    std::string names_path;        // 模型识别名称路径
    int max_batch_size;            // 批处理大小
    float conf_threshold;          // 置信度阈值
    float nms_threshold;           // 非极大值抑制阈值
    int gpu_id      = 0;           // GPU ID
    bool auto_slice = false;       // 是否自动切分
    int slice_width;               // 切片长度
    int slice_height;              // 切片高度
    double slice_horizontal_ratio; // 切片水平比例
    double slice_vertical_ratio;   // 切片垂直比例
};

class OsdConfigData : public BaseData
{
  public:
    explicit OsdConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~OsdConfigData() override = default;

  public:
    bool show_final_result    = true;  // 是否显示最终结果
    bool show_original_result = false; // 是否显示原始结果
    int font_size             = 1;     // 字体大小
};


// 用于表示 GStreamer 管道中的一个元素及其属性
struct GstPipelineElement {
  std::string element_type; // 例如: "appsrc", "queue", "video/x-raw,format=I420"
  std::map<std::string, std::string> properties; // 属性键值对, e.g., {"name", "appsrc"}, {"bitrate", "8000"}
                                               // 注意：所有属性值都存储为字符串，与GStreamer管道字符串格式一致

  // 构造函数 (可选, 但方便)
  GstPipelineElement(std::string type = "") : element_type(std::move(type)) {}
  GstPipelineElement(std::string type, std::map<std::string, std::string> props)
      : element_type(std::move(type)), properties(std::move(props)) {}
};

// 推流配置数据
// 主要用于推流配置，包含推流地址、推流协议等信息
class RecordConfigData : public BaseData
{
  public:
    explicit RecordConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~RecordConfigData() override = default;
  public:
    std::vector<GstPipelineElement> pipeline_elements;

    std::string build_gst_pipeline_string() const 
    {
      if (pipeline_elements.empty()) 
      {
          return "";
      }

      std::stringstream ss;
      for (size_t i = 0; i < pipeline_elements.size(); ++i) 
      {
          const auto& element_config = pipeline_elements[i];
          ss << element_config.element_type; // 添加元素类型或 caps 字符串

          // 添加属性
          if (!element_config.properties.empty()) 
          {
              for (const auto& prop_pair : element_config.properties) 
              {
                  ss << " " << prop_pair.first << "=" << prop_pair.second;
              }
          }

          // 如果不是最后一个元素，添加 "!" 连接符
          if (i < pipeline_elements.size() - 1) 
          {
              ss << " ! ";
          }
      }
      return ss.str();
  }
};

// 通过HTTP接口配置数据
// 主要用于通过HTTP接口配置数据，包含接口地址、请求方式等信息
class HttpConfigData : public BaseData
{
  public:
    explicit HttpConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~HttpConfigData() override = default;
};

// 目标跟踪配置数据
// 主要用于目标跟踪配置，包含跟踪算法参数等信息
class TrackingConfigData : public BaseData
{
  public:
    explicit TrackingConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~TrackingConfigData() override = default;

  public:
    int frame_rate;          // 帧率
    int track_buffer;        // 跟踪缓冲区大小
    std::string track_label; // 跟踪标签
};

// 分析配置数据
// 主要用于分析配置，包含分析算法参数等信息
class AnalyzeConfigData : public BaseData
{
  public:
    explicit AnalyzeConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~AnalyzeConfigData() override = default;
  public:
    std::string task_name; // 任务名称
    std::vector<std::vector<std::tuple<float, float>>> fences; // 围栏
};

// 路由节点配置文件
class RouterConfigData : public BaseData
{
  public:
    explicit RouterConfigData(int max_pop_batch_size = 1) : BaseData(max_pop_batch_size) {}
    ~RouterConfigData() override = default;
};

using ConfigDataPtr           = std::shared_ptr<BaseData>;
using RouterConfigDataPtr     = std::shared_ptr<RouterConfigData>;
using StreamConfigDataPtr     = std::shared_ptr<StreamConfigData>;
using OsdConfigDataPtr        = std::shared_ptr<OsdConfigData>;
using RecordConfigDataPtr     = std::shared_ptr<RecordConfigData>;
using InferConfigDataPtr      = std::shared_ptr<InferConfigData>;
using HttpConfigDataPtr       = std::shared_ptr<HttpConfigData>;
using TrackingConfigDataPtr   = std::shared_ptr<TrackingConfigData>;
using AnalyzeConfigDataPtr    = std::shared_ptr<AnalyzeConfigData>;

} // namespace common

#endif // CONFIG_HPP__