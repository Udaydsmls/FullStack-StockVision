#include <cassert>
#include <cmath>
#include <iostream>

#include "csv_loader.h"
#include "features.h"

int main() {
    stockvision::PriceTable prices;
    prices.dates = {"d1", "d2", "d3", "d4", "d5"};
    prices.open = {10, 11, 12, 13, 14};
    prices.high = {11, 12, 13, 14, 15};
    prices.low = {9, 10, 11, 12, 13};
    prices.close = {10, 11, 12, 13, 14};
    prices.volume = {1000, 1100, 1200, 1300, 1400};

    const auto features = stockvision::build_features(prices);
    assert(features.rows.size() == 5);
    assert(features.names.size() == 17);

    // A model fed a NaN returns a NaN, so the feature builder must never produce one.
    for (const auto& row : features.rows) {
        assert(row.size() == 17);
        for (const float value : row) assert(!std::isnan(value));
    }

    std::cout << "features OK\n";
    return 0;
}
