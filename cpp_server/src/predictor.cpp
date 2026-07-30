#include "predictor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace stockvision {
namespace {

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(text);
    while (std::getline(stream, part, separator)) parts.push_back(part);
    return parts;
}

std::vector<float> parse_floats(const std::string& text) {
    std::vector<float> values;
    for (const auto& part : split(text, ',')) values.push_back(std::stof(part));
    return values;
}

std::string to_upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return text;
}

// params.txt is one "KEY value" per line, written by stockvision train.
Model read_params(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read " + path.string());

    Model model;
    std::string line;
    while (std::getline(file, line)) {
        const auto space = line.find(' ');
        if (space == std::string::npos) continue;
        const std::string key = line.substr(0, space);
        const std::string value = line.substr(space + 1);

        if (key == "WINDOW") model.window = std::stoi(value);
        else if (key == "NUM_FEATURES") model.num_features = std::stoi(value);
        else if (key == "INPUT_NAME") model.input_name = value;
        else if (key == "OUTPUT_NAME") model.output_name = value;
        else if (key == "FEATURE_NAMES") model.feature_names = split(value, ',');
        else if (key == "FEATURE_MEAN") model.feature_mean = parse_floats(value);
        else if (key == "FEATURE_SCALE") model.feature_scale = parse_floats(value);
        else if (key == "TARGET_MEAN") model.target_mean = std::stof(value);
        else if (key == "TARGET_SCALE") model.target_scale = std::stof(value);
    }

    const auto expected = static_cast<std::size_t>(model.num_features);
    if (model.feature_names.size() != expected || model.feature_mean.size() != expected ||
        model.feature_scale.size() != expected) {
        throw std::runtime_error("Feature and scaler lengths disagree in " + path.string());
    }
    return model;
}

}  // namespace

const Model& Predictor::load(const std::string& ticker, const std::string& model_name) {
    const std::string key = to_upper(ticker) + "/" + model_name;

    // Building an ONNX session is slow, so hold onto it for the next request.
    std::lock_guard<std::mutex> lock(mutex_);
    const auto cached = models_.find(key);
    if (cached != models_.end()) return cached->second;

    const auto directory = artifacts_dir_ / to_upper(ticker) / model_name;
    const auto onnx_path = directory / "model.onnx";
    if (!std::filesystem::exists(onnx_path)) {
        throw std::runtime_error("No trained model for " + key + ". Run: stockvision train " +
                                 ticker + " --model " + model_name);
    }

    Model model = read_params(directory / "params.txt");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // path::c_str() is already the character type ONNX Runtime wants on each platform.
    model.session = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), options);

    std::cerr << "Loaded model " << key << "\n";
    return models_.emplace(key, std::move(model)).first->second;
}

float Predictor::predict(const std::string& ticker, const std::string& model_name,
                         const FeatureMatrix& features) {
    const Model& model = load(ticker, model_name);

    if (model.feature_names != features.names) {
        throw std::runtime_error("The trained model expects different features than we computed");
    }
    if (static_cast<int>(features.rows.size()) < model.window) {
        throw std::runtime_error("Need at least " + std::to_string(model.window) + " days of data");
    }

    // Standardise the last `window` days with the scaler saved during training.
    std::vector<float> input;
    input.reserve(static_cast<std::size_t>(model.window) * model.num_features);
    for (std::size_t day = features.rows.size() - model.window; day < features.rows.size(); ++day) {
        for (int i = 0; i < model.num_features; ++i) {
            const float scale = model.feature_scale[i] == 0.0f ? 1.0f : model.feature_scale[i];
            input.push_back((features.rows[day][i] - model.feature_mean[i]) / scale);
        }
    }

    const std::array<int64_t, 3> shape{1, model.window, model.num_features};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info_, input.data(), input.size(),
                                                        shape.data(), shape.size());

    const char* input_names[] = {model.input_name.c_str()};
    const char* output_names[] = {model.output_name.c_str()};
    auto outputs = model.session->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                      output_names, 1);

    const float scaled = outputs.front().GetTensorMutableData<float>()[0];
    return scaled * model.target_scale + model.target_mean;
}

}  // namespace stockvision
