#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "config.h"
#include "data_repository.h"
#include "http_server.h"
#include "logger.h"
#include "predictor.h"

namespace {

stockvision::LogLevel parse_log_level(const std::string& s) {
    if (s == "debug") return stockvision::LogLevel::Debug;
    if (s == "warn"  || s == "warning") return stockvision::LogLevel::Warn;
    if (s == "error") return stockvision::LogLevel::Error;
    return stockvision::LogLevel::Info;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace stockvision;

    try {
        const auto config = parse_config(argc, argv);
        Logger::instance().set_level(parse_log_level(config.log_level));

        const std::filesystem::path data_dir =
            std::getenv("STOCKVISION_DATA_DIR") ? std::getenv("STOCKVISION_DATA_DIR")
                                                : (config.artifacts_dir.parent_path() / "data");

        SV_LOG(Info, "Artifacts dir: " << config.artifacts_dir);
        SV_LOG(Info, "Data dir:      " << data_dir);

        auto repo = std::make_unique<CsvDirectoryRepository>(data_dir);
        auto predictor = std::make_unique<Predictor>(config.artifacts_dir);

        HttpServer server(config, std::move(repo), std::move(predictor));
        server.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
