#ifndef PIPELINEBUILDER_HPP__
#define PIPELINEBUILDER_HPP__

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "nodes/analyze/analyze.hpp"
#include "yaml-cpp/yaml.h" // For YAML parsing

// Include all necessary node and config headers
#include "common/config.hpp"
#include "common/data.hpp" // Assuming BaseNodeConfigData is here or in config.hpp
#include "nodes/base.hpp"
#include "nodes/infer/infer.hpp"
#include "nodes/osd/osd.hpp"
#include "nodes/record/record.hpp"
#include "nodes/router/router.hpp"
#include "nodes/stream/stream.hpp"
#include "nodes/track/track.hpp"


// Forward declare plog for logging
#include "plog/Log.h"

namespace YAML {
    template <>
    struct convert<std::tuple<float, float>> {
        static Node encode(const std::tuple<float, float>& rhs) {
            Node node;
            node.push_back(std::get<0>(rhs));
            node.push_back(std::get<1>(rhs));
            return node;
        }
    
        static bool decode(const Node& node, std::tuple<float, float>& rhs) {
            if (!node.IsSequence() || node.size() != 2) {
                return false;
            }
            try {
                rhs = std::make_tuple(node[0].as<float>(), node[1].as<float>());
            } catch (const YAML::Exception& e) {
                // PLOG_ERROR << "Failed to convert YAML node to std::tuple<float, float>: " << e.what();
                return false; // Or rethrow, or handle more gracefully
            }
            return true;
        }
    };
} // namespace YAML

class PipelineBuilder {
public:
    PipelineBuilder() = default;
    ~PipelineBuilder();

    bool load_config(const std::string& yaml_config_path);
    bool build_pipelines();
    void start_all_nodes();
    void wait_for_shutdown(); // Replaces getchar()

private:
    YAML::Node root_config_;
    std::map<std::string, std::shared_ptr<node::BaseNode>> created_nodes_;
    std::vector<std::shared_ptr<node::BaseNode>> nodes_to_start_; // To manage start order or just all unique nodes

    // Helper methods for parsing specific node configurations
    common::StreamConfigDataPtr parse_stream_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    common::InferConfigDataPtr parse_infer_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    common::OsdConfigDataPtr parse_osd_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    common::RecordConfigDataPtr parse_record_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    common::RouterConfigDataPtr parse_router_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    common::TrackingConfigDataPtr parse_tracker_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    common::AnalyzeConfigDataPtr parse_analyze_config(const YAML::Node& node_yaml, int max_pop_batch_size);
    template <typename T>
    T get_yaml_value(const YAML::Node& node, const std::string& key, const T& default_value) {
        if (node[key]) {
            try {
                return node[key].as<T>();
            } catch (const YAML::Exception& e) {
                PLOG_WARNING << "YAML parsing error for key '" << key << "': " << e.what() << ". Using default value.";
                return default_value;
            }
        }
        PLOG_DEBUG << "YAML key '" << key << "' not found. Using default value.";
        return default_value;
    }
};

#endif // PIPELINEBUILDER_HPP__