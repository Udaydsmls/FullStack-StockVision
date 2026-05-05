#include "predictor.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

#include "logger.h"

namespace stockvision {

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, delim)) out.push_back(cur);
    return out;
}

std::vector<float> parse_float_vector(const std::string& csv) {
    std::vector<float> out;
    for (const auto& tok : split(csv, ',')) {
        if (!tok.empty()) out.push_back(std::stof(tok));
    }
    return out;
}

std::filesystem::path artifact_dir(const std::filesystem::path& root,
                                   const std::string& ticker,
                                   const std::string& model) {
    std::string upper = ticker;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return root / upper / model;
}

ModelArtifact load_artifact(Ort::Env& env,
                            const std::filesystem::path& dir) {
    ModelArtifact art;
    art.input_name = "input";
    art.output_name = "output";
    art.target_mean = 0.0f;
    art.target_scale = 1.0f;

    const auto params_path = dir / "params.txt";
    std::ifstream params(params_path);
    if (!params) throw std::runtime_error("Cannot read " + params_path.string());

    std::string line;
    while (std::getline(params, line)) {
        if (line.empty()) continue;
        const auto space = line.find(' ');
        if (space == std::string::npos) continue;
        const std::string key = line.substr(0, space);
        const std::string value = line.substr(space + 1);
        if (key == "WINDOW") art.window = std::stoi(value);
        else if (key == "NUM_FEATURES") art.num_features = std::stoi(value);
        else if (key == "INPUT_NAME") art.input_name = value;
        else if (key == "OUTPUT_NAME") art.output_name = value;
        else if (key == "FEATURE_NAMES") art.feature_names = split(value, ',');
        else if (key == "FEATURE_MEAN") art.feature_mean = parse_float_vector(value);
        else if (key == "FEATURE_SCALE") art.feature_scale = parse_float_vector(value);
        else if (key == "TARGET_MEAN") art.target_mean = std::stof(value);
        else if (key == "TARGET_SCALE") art.target_scale = std::stof(value);
    }

    if (static_cast<int>(art.feature_names.size()) != art.num_features ||
        static_cast<int>(art.feature_mean.size()) != art.num_features ||
        static_cast<int>(art.feature_scale.size()) != art.num_features) {
        throw std::runtime_error("Inconsistent feature/scaler lengths in " + params_path.string());
    }

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const auto onnx_path = (dir / "model.onnx").string();
    art.session = std::make_unique<Ort::Session>(env, onnx_path.c_str(), opts);
    return art;
}

}  // namespace

std::size_t Predictor::KeyHash::operator()(const Key& k) const noexcept {
    return std::hash<std::string>{}(k.first) ^ (std::hash<std::string>{}(k.second) << 1);
}

Predictor::Predictor(std::filesystem::path artifacts_dir)
    : artifacts_dir_(std::move(artifacts_dir)) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "stockvision");
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault));
}

Predictor::~Predictor() = default;

ModelArtifact* Predictor::load(const std::string& ticker, const std::string& model) {
    const Key key{ticker, model};
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = cache_.find(key);
        if (it != cache_.end()) return it->second.get();
    }

    auto dir = artifact_dir(artifacts_dir_, ticker, model);
    if (!std::filesystem::exists(dir / "model.onnx")) {
        throw std::runtime_error("Model not trained for " + ticker + "/" + model +
                                 ". Train first with: stockvision train " + ticker + " --model " + model);
    }

    auto artifact = std::make_unique<ModelArtifact>(load_artifact(*env_, dir));

    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = cache_.emplace(key, std::move(artifact));
    SV_LOG(Info, "Loaded artifact " << ticker << "/" << model);
    return it->second.get();
}

PredictionOutput Predictor::predict(const PredictionInput& input,
                                    const FeatureMatrix& features,
                                    const std::vector<float>& closes,
                                    const std::vector<std::string>& dates,
                                    int history_size) {
    if (closes.empty()) throw std::runtime_error("Empty close-price series");

    auto* art = load(input.ticker, input.model);

    if (art->feature_names != features.column_names) {
        throw std::runtime_error("Feature schema mismatch between trained model and runtime features");
    }
    if (static_cast<int>(features.num_rows()) < art->window) {
        throw std::runtime_error("Not enough rows for window: need " + std::to_string(art->window));
    }

    const std::size_t start = features.num_rows() - art->window;
    std::vector<float> tensor;
    tensor.reserve(static_cast<std::size_t>(art->window) * art->num_features);
    for (std::size_t i = start; i < features.num_rows(); ++i) {
        for (int j = 0; j < art->num_features; ++j) {
            const float v = features.rows[i][j];
            const float scale = art->feature_scale[j] == 0.0f ? 1.0f : art->feature_scale[j];
            tensor.push_back((v - art->feature_mean[j]) / scale);
        }
    }

    const std::array<int64_t, 3> dims{1, art->window, art->num_features};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, tensor.data(), tensor.size(), dims.data(), dims.size());

    const char* input_names[]  = { art->input_name.c_str() };
    const char* output_names[] = { art->output_name.c_str() };

    auto outputs = art->session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    const float scaled = outputs.front().GetTensorMutableData<float>()[0];
    const float prediction = scaled * art->target_scale + art->target_mean;

    PredictionOutput out;
    out.prediction = prediction;
    out.last_close = closes.back();
    out.ticker = input.ticker;
    out.model = input.model;

    const int n = static_cast<int>(closes.size());
    const int take = std::min(n, history_size);
    out.history.assign(closes.end() - take, closes.end());
    if (static_cast<int>(dates.size()) >= take) {
        out.history_dates.assign(dates.end() - take, dates.end());
    }
    return out;
}

}  // namespace stockvision
