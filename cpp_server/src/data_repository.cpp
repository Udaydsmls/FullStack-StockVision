#include "data_repository.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace stockvision {

CsvDirectoryRepository::CsvDirectoryRepository(std::filesystem::path data_dir)
    : data_dir_(std::move(data_dir)) {}

std::filesystem::path CsvDirectoryRepository::csv_path(const std::string& ticker) const {
    std::string upper = ticker;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return data_dir_ / (upper + ".csv");
}

OhlcvFrame CsvDirectoryRepository::load(const std::string& ticker) {
    const auto path = csv_path(ticker);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "No CSV for '" + ticker + "' at " + path.string() +
            ". Run: stockvision fetch " + ticker);
    }
    return load_ohlcv_csv(path);
}

}  // namespace stockvision
