#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace stockvision {

struct OhlcvFrame {
    std::vector<std::string> dates;
    std::unordered_map<std::string, std::vector<double>> columns;

    [[nodiscard]] std::size_t size() const {
        if (columns.empty()) return 0;
        return columns.begin()->second.size();
    }

    [[nodiscard]] const std::vector<double>& column(const std::string& name) const;
};

OhlcvFrame load_ohlcv_csv(const std::filesystem::path& path);

}  // namespace stockvision
