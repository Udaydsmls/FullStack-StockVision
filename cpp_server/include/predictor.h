#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "feature_engineer.h"

namespace Ort { class Env; class Session; class MemoryInfo; }

namespace stockvision {

struct PredictionInput {
    std::string ticker;
    std::string model;
};

struct PredictionOutput {
    float prediction;
    float last_close;
    std::vector<float> history;
    std::vector<std::string> history_dates;
    std::string ticker;
    std::string model;
};

struct ModelArtifact {
    int window;
    int num_features;
    std::vector<std::string> feature_names;
    std::vector<float> feature_mean;
    std::vector<float> feature_scale;
    float target_mean;
    float target_scale;
    std::string input_name;
    std::string output_name;
    std::unique_ptr<Ort::Session> session;
};

class Predictor {
public:
    explicit Predictor(std::filesystem::path artifacts_dir);
    ~Predictor();

    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;

    PredictionOutput predict(const PredictionInput& input,
                             const FeatureMatrix& features,
                             const std::vector<float>& closes,
                             const std::vector<std::string>& dates,
                             int history_size);

private:
    using Key = std::pair<std::string, std::string>;
    struct KeyHash { std::size_t operator()(const Key& k) const noexcept; };

    ModelArtifact* load(const std::string& ticker, const std::string& model);

    std::filesystem::path artifacts_dir_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::unordered_map<Key, std::unique_ptr<ModelArtifact>, KeyHash> cache_;
    std::mutex mu_;
};

}  // namespace stockvision
