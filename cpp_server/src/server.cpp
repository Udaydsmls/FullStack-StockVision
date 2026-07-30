#include "server.h"

#include "httplib.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "csv_loader.h"
#include "features.h"
#include "predictor.h"

namespace stockvision {
namespace {

// --- Just enough JSON to write the responses this server returns ---

std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string list(const std::vector<float>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += number(values[i]);
    }
    return out + "]";
}

std::string list(const std::vector<std::string>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += quoted(values[i]);
    }
    return out + "]";
}

using Field = std::pair<std::string, std::string>;

std::string object(const std::vector<Field>& fields) {
    std::string out = "{";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i) out += ',';
        out += quoted(fields[i].first) + ":" + fields[i].second;
    }
    return out + "}";
}

// --- Requests and responses ---

// The browser calls us from a different port, so every response needs CORS headers.
void add_cors_headers(httplib::Response& response) {
    response.set_header("Access-Control-Allow-Origin", "*");
    response.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
    response.set_header("Access-Control-Allow-Headers", "Content-Type");
}

void send_json(httplib::Response& response, const std::string& body, int status = 200) {
    add_cors_headers(response);
    response.status = status;
    response.set_content(body, "application/json");
}

void send_error(httplib::Response& response, int status, const std::string& message) {
    std::cerr << "Request failed: " << message << "\n";
    send_json(response, object({{"detail", quoted(message)}}), status);
}

std::string string_param(const httplib::Request& request, const std::string& key,
                         const std::string& fallback) {
    const auto found = request.params.find(key);
    return found == request.params.end() ? fallback : found->second;
}

int int_param(const httplib::Request& request, const std::string& key, int fallback) {
    try {
        return std::stoi(string_param(request, key, ""));
    } catch (const std::exception&) {
        return fallback;
    }
}

// Tickers go back to the client in the same upper case the Python backend uses.
std::string ticker_param(const httplib::Request& request) {
    std::string ticker = string_param(request, "ticker", "");
    std::transform(ticker.begin(), ticker.end(), ticker.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return ticker;
}

// The last `days` closes and their dates, which is what the frontend charts.
struct History {
    std::vector<float> closes;
    std::vector<std::string> dates;
};

History take_history(const PriceTable& prices, int days) {
    const auto count = std::min(prices.size(), static_cast<std::size_t>(std::max(days, 1)));
    History history;
    history.closes.assign(prices.close.end() - count, prices.close.end());
    history.dates.assign(prices.dates.end() - count, prices.dates.end());
    return history;
}

}  // namespace

void run_server(const Settings& settings) {
    Predictor predictor(settings.artifacts_dir);
    httplib::Server server;

    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& response) {
        add_cors_headers(response);
        response.status = 204;
    });

    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        send_json(response, object({{"status", quoted("ok")}}));
    });

    server.Get("/history", [&](const httplib::Request& request, httplib::Response& response) {
        try {
            const auto ticker = ticker_param(request);
            if (ticker.empty()) return send_error(response, 400, "ticker is required");

            const auto prices = load_prices(settings.data_dir, ticker);
            const auto history =
                take_history(prices, int_param(request, "days", settings.history_days));

            send_json(response, object({
                {"ticker", quoted(ticker)},
                {"last_close", number(history.closes.back())},
                {"history", list(history.closes)},
                {"history_dates", list(history.dates)},
            }));
        } catch (const std::exception& error) {
            send_error(response, 500, error.what());
        }
    });

    server.Get("/predict", [&](const httplib::Request& request, httplib::Response& response) {
        try {
            const auto ticker = ticker_param(request);
            if (ticker.empty()) return send_error(response, 400, "ticker is required");
            const auto model = string_param(request, "model", settings.default_model);

            const auto prices = load_prices(settings.data_dir, ticker);
            const float prediction = predictor.predict(ticker, model, build_features(prices));
            const auto history =
                take_history(prices, int_param(request, "days", settings.history_days));

            send_json(response, object({
                {"ticker", quoted(ticker)},
                {"model", quoted(model)},
                {"prediction", number(prediction)},
                {"last_close", number(history.closes.back())},
                {"history", list(history.closes)},
                {"history_dates", list(history.dates)},
            }));
        } catch (const std::exception& error) {
            send_error(response, 500, error.what());
        }
    });

    std::cerr << "Listening on " << settings.host << ":" << settings.port << "\n";
    if (!server.listen(settings.host, settings.port)) {
        throw std::runtime_error("Could not bind to " + settings.host + ":" +
                                 std::to_string(settings.port));
    }
}

}  // namespace stockvision
