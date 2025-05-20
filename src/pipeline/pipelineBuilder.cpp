#include "pipeline/pipelineBuilder.hpp"
#include <fstream> // For checking file existence
#include <algorithm> // For std::find

PipelineBuilder::~PipelineBuilder() {
    PLOG_INFO << "PipelineBuilder destroyed.";
}

bool PipelineBuilder::load_config(const std::string& yaml_config_path) 
{
    std::ifstream f(yaml_config_path.c_str());
    if (!f.good()) 
    {
        PLOG_ERROR << "YAML config file not found: " << yaml_config_path;
        return false;
    }
    try 
    {
        root_config_ = YAML::LoadFile(yaml_config_path);
        PLOG_INFO << "Successfully loaded YAML config: " << yaml_config_path;
    } 
    catch (const YAML::Exception& e) 
    {
        PLOG_ERROR << "Error parsing YAML file " << yaml_config_path << ": " << e.what();
        return false;
    }
    return true;
}

// --- Configuration Parsers ---
common::StreamConfigDataPtr PipelineBuilder::parse_stream_config(const YAML::Node& node_yaml, int max_pop_batch_size) 
{
    auto config_yaml = node_yaml["config"];
    if (!config_yaml) 
    {
        PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>() << "' missing 'config' section.";
        return nullptr;
    }
    // The max_pop_batch_size from the node level is passed to the config constructor
    auto stream_config = std::make_shared<common::StreamConfigData>(max_pop_batch_size);

    stream_config->gpu_id             = get_yaml_value(config_yaml, "gpu_id", 0);
    stream_config->decode_type        = get_yaml_value<std::string>(config_yaml, "decode_type", "GPU");
    stream_config->skip_frame         = get_yaml_value(config_yaml, "skip_frame", 0);
    stream_config->stream_url         = get_yaml_value<std::string>(config_yaml, "stream_url", "");
    stream_config->owner_pipeline_id  = get_yaml_value<std::string>(config_yaml, "owner_pipeline_id", "");
    stream_config->stream_name        = get_yaml_value<std::string>(config_yaml, "stream_name", node_yaml["name"].as<std::string>());

    if (stream_config->stream_url.empty()) {
        PLOG_ERROR << "Stream node '" << node_yaml["name"].as<std::string>() << "' missing 'stream_url'.";
        return nullptr;
    }
    return stream_config;
}

common::InferConfigDataPtr PipelineBuilder::parse_infer_config(const YAML::Node& node_yaml, int max_pop_batch_size) {
    auto config_yaml = node_yaml["config"];
    if (!config_yaml) 
    {
        PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>() << "' missing 'config' section.";
        return nullptr;
    }
    auto infer_config = std::make_shared<common::InferConfigData>(max_pop_batch_size);

    infer_config->model_path = get_yaml_value<std::string>(config_yaml, "model_path", "");
    infer_config->model_type = get_yaml_value<std::string>(config_yaml, "model_type", "");
    infer_config->names_path = get_yaml_value<std::string>(config_yaml, "names_path", "");
    infer_config->max_batch_size = get_yaml_value(config_yaml, "max_batch_size", 1);
    infer_config->gpu_id = get_yaml_value(config_yaml, "gpu_id", 0);
    infer_config->conf_threshold = get_yaml_value(config_yaml, "conf_threshold", 0.25f);
    infer_config->nms_threshold = get_yaml_value(config_yaml, "nms_threshold", 0.45f);
    infer_config->auto_slice = get_yaml_value(config_yaml, "auto_slice", true);
    infer_config->slice_width = get_yaml_value(config_yaml, "slice_width", 0);
    infer_config->slice_height = get_yaml_value(config_yaml, "slice_height", 0);
    infer_config->slice_horizontal_ratio = get_yaml_value(config_yaml, "slice_horizontal_ratio", 0.0);
    infer_config->slice_vertical_ratio = get_yaml_value(config_yaml, "slice_vertical_ratio", 0.0);
    
    if (infer_config->model_path.empty()) {
         PLOG_ERROR << "Infer node '" << node_yaml["name"].as<std::string>() << "' missing 'model_path'.";
        return nullptr;
    }
    if (infer_config->names_path.empty()) {
        PLOG_ERROR << "Infer node '" << node_yaml["name"].as<std::string>() << "' missing 'names_path'.";
        return nullptr;
    }
    return infer_config;
}

common::OsdConfigDataPtr PipelineBuilder::parse_osd_config(const YAML::Node& node_yaml, int max_pop_batch_size) {
    auto config_yaml = node_yaml["config"];
    // OSD config might be optional or have all defaults
    auto osd_config = std::make_shared<common::OsdConfigData>(max_pop_batch_size);
    if (config_yaml) 
    {
        osd_config->show_final_result = get_yaml_value(config_yaml, "show_final_result", false);
        osd_config->show_original_result = get_yaml_value(config_yaml, "show_original_result", true);
    } 
    else 
    {
        PLOG_WARNING << "Node '" << node_yaml["name"].as<std::string>() << "' (OSD) missing 'config' section. Using defaults.";
        // Set defaults explicitly if constructor doesn't
        osd_config->show_final_result = false;
        osd_config->show_original_result = true;
    }
    return osd_config;
}

common::RecordConfigDataPtr PipelineBuilder::parse_record_config(const YAML::Node& node_yaml, int max_pop_batch_size) 
{
    // 1. Get the 'config' node, which is now expected to be a sequence.
    YAML::Node config_elements_node = node_yaml["config"];
    if (!config_elements_node) 
    {
        PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>() << "' missing 'config' section." << std::endl;
        return nullptr;
    }

    // 2. Verify that 'config' is a sequence (list).
    if (!config_elements_node.IsSequence()) 
    {
        PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>()
                   << "': 'config' section is not a list of pipeline elements." << std::endl;
        return nullptr;
    }

    auto record_config = std::make_shared<common::RecordConfigData>(max_pop_batch_size);

    // 3. Iterate through each element in the 'config' sequence.
    for (const auto& element_item_node : config_elements_node) 
    { // yaml-cpp sequence iteration
        if (!element_item_node.IsMap()) 
        {
            PLOG_WARNING << "Node '" << node_yaml["name"].as<std::string>()
                         << "': Found an item in 'config' that is not a map (element definition). Skipping." << std::endl;
            continue;
        }

        common::GstPipelineElement current_gst_element;

        // 4. Parse the 'element' type/string for the current GStreamer element.
        YAML::Node type_node = element_item_node["element"];
        if (type_node && type_node.IsScalar()) 
        {
            current_gst_element.element_type = type_node.as<std::string>();
        } 
        else 
        {
            PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>()
                       << "': A pipeline element is missing 'element' field or it's not a string." << std::endl;
            return nullptr;
        }

        YAML::Node properties_node = element_item_node["properties"];
        if (properties_node && properties_node.IsMap()) 
        {
            for (YAML::const_iterator it = properties_node.begin(); it != properties_node.end(); ++it) 
            {
                YAML::Node key_node = it->first;   // Access the key node
                YAML::Node value_node = it->second; // Access the value node

                if (key_node.IsScalar() && value_node.IsScalar()) 
                {
                    std::string key = key_node.as<std::string>();
                    std::string value_str = value_node.as<std::string>();
                    current_gst_element.properties[key] = value_str;
                } 
                else 
                {
                    PLOG_WARNING << "Node '" << node_yaml["name"].as<std::string>()
                                 << "': Property key or value is not a scalar for element '"
                                 << current_gst_element.element_type << "'. Skipping property." << std::endl;
                }
            }
        }
        record_config->pipeline_elements.push_back(current_gst_element);
    }

    // 6. Check if any elements were successfully parsed.
    if (record_config->pipeline_elements.empty()) 
    {
        PLOG_ERROR << "Record node '" << node_yaml["name"].as<std::string>()
                   << "' resulted in an empty pipeline (e.g., 'config: []' or all elements malformed/skipped)." << std::endl;
        return nullptr; // Consider an empty pipeline an error, similar to missing gst_pipeline string.
    }
    return record_config;
}

common::TrackingConfigDataPtr PipelineBuilder::parse_tracker_config(const YAML::Node& node_yaml, int max_pop_batch_size) 
{
    auto config_yaml = node_yaml["config"];
    if (!config_yaml) 
    {
        PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>() << "' missing 'config' section.";
        return nullptr;
    }
    auto tracker_config = std::make_shared<common::TrackingConfigData>(max_pop_batch_size);
    tracker_config->frame_rate = get_yaml_value(config_yaml, "frame_rate", 30);
    tracker_config->track_buffer = get_yaml_value(config_yaml, "track_buffer", 30);
    tracker_config->track_label = get_yaml_value<std::string>(config_yaml, "track_label", "");

    if (tracker_config->track_label.empty()) 
    {
        PLOG_ERROR << "Tracker node '" << node_yaml["name"].as<std::string>() << "' missing 'track_label'.";
        return nullptr;
    }
    return tracker_config;
}

common::RouterConfigDataPtr PipelineBuilder::parse_router_config(const YAML::Node& node_yaml, int max_pop_batch_size) 
{
    int router_specific_param = max_pop_batch_size; // Defaulting to this for now.
    auto config_yaml = node_yaml["config"];
    if (config_yaml && config_yaml["num_outputs"]) 
    {
        router_specific_param = config_yaml["num_outputs"].as<int>();
    } 
    else 
    {
        PLOG_WARNING << "Router node '" << node_yaml["name"].as<std::string>() 
                     << "' missing specific config (e.g. num_outputs). Using max_pop_batch_size (" 
                     << max_pop_batch_size << ") for its config constructor.";
    }

    auto router_config = std::make_shared<common::RouterConfigData>(router_specific_param);
    return router_config;
}

/**
  - name: person_count
    type: ANALYZE
    max_pop_batch_size: 4
    config:
      task_name: person_count
      fences: 
        - [[100, 100], [200, 100], [200, 200], [100, 200]]
        - [[300, 300], [400, 300], [400, 400], [300, 400]]
*/
common::AnalyzeConfigDataPtr PipelineBuilder::parse_analyze_config(const YAML::Node& node_yaml, int max_pop_batch_size)
{
    auto config_yaml = node_yaml["config"];
    if (!config_yaml) 
    {
        PLOG_ERROR << "Node '" << node_yaml["name"].as<std::string>() << "' missing 'config' section.";
        return nullptr;
    }
    auto analyze_config = std::make_shared<common::AnalyzeConfigData>(max_pop_batch_size);
    analyze_config->task_name = get_yaml_value<std::string>(config_yaml, "task_name", "");
    analyze_config->fences = get_yaml_value<std::vector<std::vector<std::tuple<float, float>>>>(config_yaml, "fences", {});

    if (analyze_config->task_name.empty()) 
    {
        PLOG_ERROR << "Analyze node '" << node_yaml["name"].as<std::string>() << "' missing 'task_name'.";
        return nullptr;
    }
    return analyze_config;
}


bool PipelineBuilder::build_pipelines() 
{
    if (!root_config_["nodes"]) 
    {
        PLOG_ERROR << "YAML missing 'nodes' section.";
        return false;
    }
    if (!root_config_["pipelines"]) 
    {
        PLOG_ERROR << "YAML missing 'pipelines' section.";
        return false;
    }

    // 1. Create all nodes
    PLOG_INFO << "--- Creating Nodes ---";
    const YAML::Node& nodes_yaml = root_config_["nodes"];
    for (const auto& node_entry : nodes_yaml) 
    {
        std::string node_name     = get_yaml_value<std::string>(node_entry, "name", "");
        std::string node_type_str = get_yaml_value<std::string>(node_entry, "type", "");
        int max_pop_batch_size    = get_yaml_value(node_entry, "max_pop_batch_size", 1);

        if (node_name.empty() || node_type_str.empty()) 
        {
            PLOG_ERROR << "Node entry missing 'name' or 'type'. Skipping.";
            continue;
        }

        if (created_nodes_.count(node_name)) 
        {
            PLOG_ERROR << "Duplicate node name '" << node_name << "' defined. Skipping.";
            continue;
        }

        std::shared_ptr<node::BaseNode> current_node = nullptr;
        PLOG_INFO << "Creating node: " << node_name << " (Type: " << node_type_str << ", MaxPop: " << max_pop_batch_size << ")";

        if (node_type_str == "STREAM") 
        {
            auto config = parse_stream_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::StreamNode>(node_name, config);
        } 
        else if (node_type_str == "INFER") 
        {
            auto config = parse_infer_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::InferNode>(node_name, config);
        } 
        else if (node_type_str == "OSD") 
        {
            auto config = parse_osd_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::OsdNode>(node_name, config);
        } 
        else if (node_type_str == "RECORD") 
        {
            auto config = parse_record_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::RecordNode>(node_name, config);
        } 
        else if (node_type_str == "ROUTER") 
        {
            auto config = parse_router_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::RouterNode>(node_name, config);
        }
        else if (node_type_str == "TRACKER") 
        {
            auto config = parse_tracker_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::TrackNode>(node_name, config);
        }
        else if (node_type_str == "ANALYZE") 
        {
            auto config = parse_analyze_config(node_entry, max_pop_batch_size);
            if (config) 
                current_node = std::make_shared<node::AnalyzeNode>(node_name, config);
        }
        else 
        {
            PLOG_ERROR << "Unknown node type '" << node_type_str << "' for node '" << node_name << "'. Skipping.";
            continue;
        }

        if (current_node) 
        {
            created_nodes_[node_name] = current_node;
            if (std::find(nodes_to_start_.begin(), nodes_to_start_.end(), current_node) == nodes_to_start_.end()) 
            {
                nodes_to_start_.push_back(current_node);
            }
        } 
        else 
        {
            PLOG_ERROR << "Failed to create node '" << node_name << "'. Configuration might be invalid.";
            return false; // Critical error
        }
    }

    // 2. Link nodes according to pipelines
    PLOG_INFO << "--- Linking Pipelines ---";
    const YAML::Node& pipelines_yaml = root_config_["pipelines"];
    for (const auto& pipeline_entry : pipelines_yaml) 
    {
        std::string pipeline_id = get_yaml_value<std::string>(pipeline_entry, "id", "");
        if (pipeline_id.empty()) 
        {
            PLOG_ERROR << "Pipeline entry missing 'id'. Skipping.";
            continue;
        }
        PLOG_INFO << "Configuring pipeline ID: " << pipeline_id;

        const YAML::Node& pipeline_nodes_yaml = pipeline_entry["nodes"];
        if (!pipeline_nodes_yaml || !pipeline_nodes_yaml.IsSequence()) 
        {
            PLOG_ERROR << "Pipeline '" << pipeline_id << "' missing 'nodes' list or it's not a sequence. Skipping.";
            continue;
        }

        std::shared_ptr<node::BaseNode> prev_node = nullptr;
        for (const auto& node_name_yaml : pipeline_nodes_yaml) 
        {
            std::string node_name = node_name_yaml.as<std::string>();
            if (created_nodes_.find(node_name) == created_nodes_.end()) 
            {
                PLOG_ERROR << "Node '" << node_name << "' referenced in pipeline '" << pipeline_id << "' not defined. Skipping pipeline.";
                prev_node = nullptr; // Break chain for this pipeline
                break;
            }
            std::shared_ptr<node::BaseNode> current_node = created_nodes_[node_name];
            if (auto stream_node = std::dynamic_pointer_cast<node::StreamNode>(current_node)) 
            {
                // Check if its config is StreamConfigData
                if(auto stream_config = std::dynamic_pointer_cast<common::StreamConfigData>(stream_node->get_config_data()))
                {
                    PLOG_DEBUG << "Setting owner_pipeline_id '" << pipeline_id << "' for StreamNode '" << node_name << "'";
                    stream_config->owner_pipeline_id = pipeline_id;
                }
            }
            if (prev_node) 
            {
                PLOG_INFO << "Linking " << prev_node->get_name() << " -> " << current_node->get_name() << " for pipeline " << pipeline_id;
                node::LinkNode(prev_node, current_node, pipeline_id);
            }
            prev_node = current_node;
        }
    }
    PLOG_INFO << "Pipeline building process completed.";
    return true;
}

void PipelineBuilder::start_all_nodes() {
    PLOG_INFO << "--- Starting Nodes ---";
    
    std::vector<std::shared_ptr<node::BaseNode>> start_order_nodes;
    PLOG_INFO << "Starting nodes in reverse order of their definition in YAML 'nodes' list.";
    for (auto it = nodes_to_start_.rbegin(); it != nodes_to_start_.rend(); ++it) 
    {
        PLOG_INFO << "Starting node: " << (*it)->get_name();
        (*it)->start();
    }
    PLOG_INFO << "All specified nodes started.";
}

void PipelineBuilder::wait_for_shutdown() {
    PLOG_INFO << "Application running. Press Enter to exit...";
    getchar(); // Or use a more sophisticated shutdown signal mechanism
    PLOG_INFO << "Shutdown signal received. Exiting.";
}