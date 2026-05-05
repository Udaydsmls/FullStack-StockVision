#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "csv_loader.h"

namespace stockvision {

struct FeatureMatrix {
    std::vector<std::string> column_names;
    std::vector<std::vector<float>> rows;  // rows[time_index][feature_index]

    [[nodiscard]] std::size_t num_rows() const { return rows.size(); }
    [[nodiscard]] std::size_t num_cols() const { return column_names.size(); }
};

FeatureMatrix build_feature_matrix(const OhlcvFrame& frame);

}  // namespace stockvision
