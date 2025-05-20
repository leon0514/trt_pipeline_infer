#include "pipeline/pipelineBuilder.hpp"
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"

int main(int argc, char* argv[]) {
    plog::init(plog::debug, "log/pipeline_builder_vsp.log", 1000000, 5); // Init plog

    std::string config_file = "demo.yaml"; // Default config file name
    if (argc > 1) {
        config_file = argv[1]; // Allow specifying config file via command line
    }

    PLOG_INFO << "Starting application with config: " << config_file;

    PipelineBuilder builder;

    if (!builder.load_config(config_file)) {
        PLOG_FATAL << "Failed to load configuration. Exiting.";
        return 1;
    }

    if (!builder.build_pipelines()) {
        PLOG_FATAL << "Failed to build pipelines. Exiting.";
        return 1;
    }

    builder.start_all_nodes();

    builder.wait_for_shutdown(); // Keeps the application running

    PLOG_INFO << "Application finished.";
    return 0;
}