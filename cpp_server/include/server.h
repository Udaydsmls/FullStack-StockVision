#pragma once

#include <filesystem>
#include <string>

namespace stockvision {

struct Settings {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::filesystem::path artifacts_dir = "artifacts";
    std::filesystem::path data_dir = "data";
    std::string default_model = "lstm";
    int history_days = 60;
};

// Serves /health, /history and /predict until the process is stopped.
void run_server(const Settings& settings);

}  // namespace stockvision
