#pragma once

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace stockvision {

enum class LogLevel { Debug, Info, Warn, Error };

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }

    void log(LogLevel level, const std::string& msg) {
        if (level < level_) return;
        std::lock_guard<std::mutex> lock(mu_);
        std::cerr << timestamp() << " | " << level_name(level) << " | " << msg << '\n';
    }

private:
    Logger() = default;

    static std::string timestamp() {
        const auto now = std::chrono::system_clock::now();
        const auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
#if defined(_WIN32)
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
        return buf;
    }

    static const char* level_name(LogLevel level) {
        switch (level) {
            case LogLevel::Debug: return "DEBUG";
            case LogLevel::Info:  return "INFO ";
            case LogLevel::Warn:  return "WARN ";
            case LogLevel::Error: return "ERROR";
        }
        return "INFO ";
    }

    LogLevel level_ = LogLevel::Info;
    std::mutex mu_;
};

#define SV_LOG(level, expr)                                                         \
    do {                                                                            \
        std::ostringstream _sv_oss;                                                 \
        _sv_oss << expr;                                                            \
        ::stockvision::Logger::instance().log(::stockvision::LogLevel::level,       \
                                              _sv_oss.str());                       \
    } while (0)

}  // namespace stockvision
