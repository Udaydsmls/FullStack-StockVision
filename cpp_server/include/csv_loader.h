#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace stockvision {

// One CSV of daily bars. Every vector holds one value per trading day.
struct PriceTable {
    std::vector<std::string> dates;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    std::size_t size() const { return close.size(); }
};

PriceTable read_csv(const std::filesystem::path& path);

// Reads data_dir/<TICKER>.csv, the file `stockvision fetch` writes.
PriceTable load_prices(const std::filesystem::path& data_dir, const std::string& ticker);

}  // namespace stockvision
