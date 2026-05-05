#pragma once

#include <filesystem>
#include <string>

#include "csv_loader.h"

namespace stockvision {

class DataRepository {
public:
    virtual ~DataRepository() = default;
    virtual OhlcvFrame load(const std::string& ticker) = 0;
};

class CsvDirectoryRepository final : public DataRepository {
public:
    explicit CsvDirectoryRepository(std::filesystem::path data_dir);
    OhlcvFrame load(const std::string& ticker) override;
    [[nodiscard]] std::filesystem::path csv_path(const std::string& ticker) const;

private:
    std::filesystem::path data_dir_;
};

}  // namespace stockvision
