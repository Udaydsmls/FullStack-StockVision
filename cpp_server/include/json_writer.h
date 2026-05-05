#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace stockvision::json {

inline std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const char ch : s) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

inline std::string number(double v) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    std::ostringstream oss;
    oss << std::setprecision(10) << v;
    return oss.str();
}

class Object {
public:
    Object& field(const std::string& key, const std::string& value) {
        comma();
        oss_ << '"' << escape(key) << "\":\"" << escape(value) << '"';
        return *this;
    }
    Object& field(const std::string& key, double value) {
        comma();
        oss_ << '"' << escape(key) << "\":" << number(value);
        return *this;
    }
    Object& field(const std::string& key, int value) {
        comma();
        oss_ << '"' << escape(key) << "\":" << value;
        return *this;
    }
    Object& array(const std::string& key, const std::vector<float>& values) {
        comma();
        oss_ << '"' << escape(key) << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) oss_ << ',';
            oss_ << number(values[i]);
        }
        oss_ << ']';
        return *this;
    }
    Object& array(const std::string& key, const std::vector<std::string>& values) {
        comma();
        oss_ << '"' << escape(key) << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) oss_ << ',';
            oss_ << '"' << escape(values[i]) << '"';
        }
        oss_ << ']';
        return *this;
    }
    [[nodiscard]] std::string str() const { return "{" + oss_.str() + "}"; }

private:
    void comma() {
        if (count_++) oss_ << ',';
    }
    std::ostringstream oss_;
    int count_ = 0;
};

}  // namespace stockvision::json
