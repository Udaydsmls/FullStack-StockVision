#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "csv_loader.h"
#include "feature_engineer.h"

int main() {
    stockvision::OhlcvFrame frame;
    frame.dates = {"d1", "d2", "d3", "d4", "d5"};
    frame.columns["open"]   = {10, 11, 12, 13, 14};
    frame.columns["high"]   = {11, 12, 13, 14, 15};
    frame.columns["low"]    = {9, 10, 11, 12, 13};
    frame.columns["close"]  = {10, 11, 12, 13, 14};
    frame.columns["volume"] = {1000, 1100, 1200, 1300, 1400};

    auto matrix = stockvision::build_feature_matrix(frame);
    assert(matrix.num_rows() == 5);
    assert(matrix.num_cols() == 17);
    for (const auto& row : matrix.rows) {
        for (const float v : row) {
            assert(!std::isnan(v));
        }
    }
    std::cout << "feature_engineer OK\n";
    return 0;
}
