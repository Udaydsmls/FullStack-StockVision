#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "csv_loader.h"

int main() {
    const auto path = std::filesystem::temp_directory_path() / "stockvision_test.csv";
    std::ofstream(path) << "Date,Close,High,Low,Open,Volume\n"
                           "2025-01-01,10.5,11.0,10.0,10.2,1000\n"
                           "2025-01-02,10.7,11.2,10.3,10.5,2000\n";

    // Columns are matched by name, so the out-of-order Close/Open still line up.
    const auto prices = stockvision::read_csv(path);
    assert(prices.size() == 2);
    assert(prices.dates.size() == 2);
    assert(prices.close[0] == 10.5);
    assert(prices.open[0] == 10.2);
    assert(prices.volume[1] == 2000);

    std::cout << "csv_loader OK\n";
    return 0;
}
