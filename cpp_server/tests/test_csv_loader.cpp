#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "csv_loader.h"

namespace fs = std::filesystem;

static fs::path write_temp(const std::string& body) {
    const auto path = fs::temp_directory_path() / "stockvision_test.csv";
    std::ofstream out(path);
    out << body;
    return path;
}

int main() {
    const auto path = write_temp(
        "Date,Close,High,Low,Open,Volume\n"
        "2025-01-01,10.5,11.0,10.0,10.2,1000\n"
        "2025-01-02,10.7,11.2,10.3,10.5,2000\n");

    auto frame = stockvision::load_ohlcv_csv(path);
    assert(frame.size() == 2);
    assert(frame.dates.size() == 2);
    assert(frame.column("close")[0] == 10.5);
    assert(frame.column("volume")[1] == 2000);
    std::cout << "csv_loader OK\n";
    return 0;
}
