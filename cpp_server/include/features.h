#pragma once

#include <string>
#include <vector>

#include "csv_loader.h"

namespace stockvision {

// The same 17 columns, in the same order, as stockvision/features.py.
struct FeatureMatrix {
    std::vector<std::string> names;
    std::vector<std::vector<float>> rows;  // rows[day][feature]
};

FeatureMatrix build_features(const PriceTable& prices);

}  // namespace stockvision
