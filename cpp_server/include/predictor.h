#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "features.h"

namespace stockvision {

// Everything params.txt records about one exported model, plus its ONNX session.
struct Model {
    int window = 0;
    int num_features = 0;
    std::vector<std::string> feature_names;
    std::vector<float> feature_mean;
    std::vector<float> feature_scale;
    float target_mean = 0.0f;
    float target_scale = 1.0f;
    std::string input_name = "input";
    std::string output_name = "output";
    std::unique_ptr<Ort::Session> session;
};

class Predictor {
public:
    explicit Predictor(std::filesystem::path artifacts_dir)
        : artifacts_dir_(std::move(artifacts_dir)) {}

    // Scales the last `window` rows, runs the model, and returns the next close in dollars.
    float predict(const std::string& ticker, const std::string& model_name,
                  const FeatureMatrix& features);

private:
    const Model& load(const std::string& ticker, const std::string& model_name);

    std::filesystem::path artifacts_dir_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "stockvision"};
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::map<std::string, Model> models_;  // keyed by "AAPL/lstm"
    std::mutex mutex_;
};

}  // namespace stockvision
