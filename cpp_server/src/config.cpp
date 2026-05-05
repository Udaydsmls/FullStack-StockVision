#include "config.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace stockvision {

void print_usage(const char* program) {
    std::cerr <<
        "Usage: " << program << " [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --host HOST             Bind address (default: 0.0.0.0).\n"
        "  --port PORT             Bind port (default: 8080).\n"
        "  --artifacts-dir DIR     Directory containing per-ticker ONNX artifacts.\n"
        "  --default-model NAME    Model used when the request does not specify one.\n"
        "  --history-size N        Number of historical bars returned in responses.\n"
        "  --log-level LEVEL       debug | info | warn | error.\n"
        "  -h, --help              Show this help text and exit.\n"
        "\n"
        "Environment overrides:\n"
        "  STOCKVISION_ARTIFACTS_DIR, STOCKVISION_DEFAULT_MODEL, STOCKVISION_HISTORY_SIZE.\n";
}

namespace {

std::string env_or(const char* key, const std::string& fallback) {
    const char* raw = std::getenv(key);
    return (raw && *raw) ? std::string(raw) : fallback;
}

int env_int_or(const char* key, int fallback) {
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return fallback;
    try {
        return std::stoi(raw);
    } catch (...) {
        return fallback;
    }
}

std::string require_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("Missing value for option ") + argv[i]);
    }
    return argv[++i];
}

}  // namespace

ServerConfig parse_config(int argc, char** argv) {
    ServerConfig cfg;
    cfg.artifacts_dir = env_or("STOCKVISION_ARTIFACTS_DIR", "artifacts");
    cfg.default_model = env_or("STOCKVISION_DEFAULT_MODEL", cfg.default_model);
    cfg.history_size = env_int_or("STOCKVISION_HISTORY_SIZE", cfg.history_size);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--host") {
            cfg.host = require_value(i, argc, argv);
        } else if (arg == "--port") {
            cfg.port = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--artifacts-dir") {
            cfg.artifacts_dir = require_value(i, argc, argv);
        } else if (arg == "--default-model") {
            cfg.default_model = require_value(i, argc, argv);
        } else if (arg == "--history-size") {
            cfg.history_size = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--log-level") {
            cfg.log_level = require_value(i, argc, argv);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (cfg.port <= 0 || cfg.port > 65535) {
        throw std::invalid_argument("Port must be in (0, 65535]");
    }
    if (cfg.history_size < 1) {
        throw std::invalid_argument("history-size must be >= 1");
    }
    return cfg;
}

}  // namespace stockvision
