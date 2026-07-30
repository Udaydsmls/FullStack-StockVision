#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "server.h"

namespace {

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [OPTIONS]\n\n"
              << "  --host HOST            Bind address (default: 0.0.0.0)\n"
              << "  --port PORT            Bind port (default: 8080)\n"
              << "  --artifacts-dir DIR    Where stockvision train wrote the ONNX models\n"
              << "  --data-dir DIR         Where the cached OHLCV CSVs live\n"
              << "  --default-model NAME   Model to use when a request omits one\n"
              << "  --history-days N       How many closes to return (default: 60)\n"
              << "  -h, --help             Show this message\n";
}

stockvision::Settings parse_arguments(int argc, char** argv) {
    stockvision::Settings settings;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
        const std::string value = argv[++i];

        if (flag == "--host") settings.host = value;
        else if (flag == "--port") settings.port = std::stoi(value);
        else if (flag == "--artifacts-dir") settings.artifacts_dir = value;
        else if (flag == "--data-dir") settings.data_dir = value;
        else if (flag == "--default-model") settings.default_model = value;
        else if (flag == "--history-days") settings.history_days = std::stoi(value);
        else throw std::invalid_argument("Unknown option: " + flag);
    }

    if (settings.port <= 0 || settings.port > 65535) {
        throw std::invalid_argument("Port must be between 1 and 65535");
    }
    return settings;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto settings = parse_arguments(argc, argv);
        std::cerr << "Artifacts: " << settings.artifacts_dir << "\n"
                  << "Data:      " << settings.data_dir << "\n";
        stockvision::run_server(settings);
    } catch (const std::exception& error) {
        std::cerr << "Fatal: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
