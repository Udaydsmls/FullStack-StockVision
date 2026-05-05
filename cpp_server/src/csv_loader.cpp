#include "csv_loader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace stockvision {

const std::vector<double>& OhlcvFrame::column(const std::string& name) const {
    const auto it = columns.find(name);
    if (it == columns.end()) {
        throw std::out_of_range("Column not present in OhlcvFrame: " + name);
    }
    return it->second;
}

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string current;
    bool inside_quotes = false;
    for (const char ch : line) {
        if (ch == '"') {
            inside_quotes = !inside_quotes;
        } else if (ch == ',' && !inside_quotes) {
            cells.push_back(current);
            current.clear();
        } else if (ch != '\r') {
            current.push_back(ch);
        }
    }
    cells.push_back(current);
    return cells;
}

}  // namespace

OhlcvFrame load_ohlcv_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open CSV file: " + path.string());
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("CSV file is empty: " + path.string());
    }
    auto headers = split_csv_line(header_line);
    std::vector<std::string> normalised;
    normalised.reserve(headers.size());
    for (const auto& h : headers) normalised.push_back(to_lower(h));

    OhlcvFrame frame;
    const auto canonical_for = [&](const std::string& lower) -> std::string {
        if (lower == "date" || lower == "datetime" || lower == "timestamp") return "date";
        if (lower == "open" || lower == "openprice") return "open";
        if (lower == "high") return "high";
        if (lower == "low") return "low";
        if (lower == "close" || lower == "adj close" || lower == "adj_close" || lower == "adjclose") {
            return "close";
        }
        if (lower == "volume" || lower == "vol") return "volume";
        return "";
    };

    std::vector<std::string> col_for_index(normalised.size());
    for (std::size_t i = 0; i < normalised.size(); ++i) {
        col_for_index[i] = canonical_for(normalised[i]);
        if (!col_for_index[i].empty() && col_for_index[i] != "date") {
            frame.columns[col_for_index[i]];
        }
    }

    bool has_date = std::find(col_for_index.begin(), col_for_index.end(), std::string("date"))
                    != col_for_index.end();

    for (const auto& required : {"open", "high", "low", "close", "volume"}) {
        if (frame.columns.find(required) == frame.columns.end()) {
            throw std::runtime_error(std::string("Required column missing in CSV: ") + required);
        }
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto cells = split_csv_line(line);
        if (cells.size() != col_for_index.size()) continue;

        for (std::size_t i = 0; i < cells.size(); ++i) {
            const auto& canonical = col_for_index[i];
            if (canonical.empty()) continue;
            if (canonical == "date") {
                frame.dates.push_back(cells[i]);
                continue;
            }
            try {
                frame.columns[canonical].push_back(std::stod(cells[i]));
            } catch (const std::exception&) {
                frame.columns[canonical].push_back(0.0);
            }
        }
    }

    if (!has_date) {
        const std::size_t n = frame.size();
        frame.dates.resize(n);
    }
    if (frame.size() == 0) {
        throw std::runtime_error("CSV had no data rows: " + path.string());
    }
    return frame;
}

}  // namespace stockvision
