#include "csv_loader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <utility>  // std::pair

namespace stockvision {
namespace {

std::string to_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text;
}

std::string to_upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return text;
}

std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    bool in_quotes = false;
    for (const char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            cells.push_back(cell);
            cell.clear();
        } else if (c != '\r') {
            cell.push_back(c);
        }
    }
    cells.push_back(cell);
    return cells;
}

// yfinance header names vary a little between versions.
std::string canonical_name(const std::string& header) {
    const std::string name = to_lower(header);
    if (name == "datetime" || name == "timestamp") return "date";
    if (name == "adj close" || name == "adj_close") return "close";
    return name;
}

double to_double(const std::string& text) {
    try {
        return std::stod(text);
    } catch (const std::exception&) {
        return 0.0;  // blanks and "null" become zero rather than killing the request
    }
}

}  // namespace

PriceTable read_csv(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open CSV file: " + path.string());

    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("CSV file is empty: " + path.string());

    PriceTable table;
    const std::vector<std::pair<std::string, std::vector<double>*>> wanted = {
        {"open", &table.open},   {"high", &table.high},     {"low", &table.low},
        {"close", &table.close}, {"volume", &table.volume},
    };

    // Work out, once, which vector each CSV column feeds. nullptr means ignore it.
    const auto headers = split_line(line);
    std::vector<std::vector<double>*> destination(headers.size(), nullptr);
    std::size_t date_column = headers.size();  // headers.size() means "no date column"

    for (std::size_t i = 0; i < headers.size(); ++i) {
        const std::string name = canonical_name(headers[i]);
        if (name == "date") date_column = i;
        for (const auto& [wanted_name, values] : wanted) {
            if (name == wanted_name) destination[i] = values;
        }
    }

    for (const auto& [name, values] : wanted) {
        if (std::find(destination.begin(), destination.end(), values) == destination.end()) {
            throw std::runtime_error("CSV has no '" + name + "' column: " + path.string());
        }
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        const auto cells = split_line(line);
        if (cells.size() != headers.size()) continue;  // skip malformed rows

        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (i == date_column) table.dates.push_back(cells[i]);
            else if (destination[i]) destination[i]->push_back(to_double(cells[i]));
        }
    }

    if (table.size() == 0) throw std::runtime_error("CSV has no data rows: " + path.string());
    table.dates.resize(table.size());  // pad with blanks if there was no date column
    return table;
}

PriceTable load_prices(const std::filesystem::path& data_dir, const std::string& ticker) {
    const auto path = data_dir / (to_upper(ticker) + ".csv");
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("No cached data for '" + ticker + "' at " + path.string() +
                                 ". Run: stockvision fetch " + ticker);
    }
    return read_csv(path);
}

}  // namespace stockvision
