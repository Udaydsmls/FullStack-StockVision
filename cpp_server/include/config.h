#pragma once

#include <filesystem>
#include <string>

namespace stockvision {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::filesystem::path artifacts_dir = "artifacts";
    std::string default_model = "lstm";
    std::string log_level = "info";
    int history_size = 60;
};

ServerConfig parse_config(int argc, char** argv);
void print_usage(const char* program);

}  // namespace stockvision
