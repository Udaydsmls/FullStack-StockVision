#include "http_server.h"

#include "httplib.h"

#include "feature_engineer.h"
#include "json_writer.h"
#include "logger.h"

#include <exception>
#include <functional>
#include <utility>

namespace stockvision {

struct HttpServer::Impl {
    ServerConfig config;
    std::unique_ptr<DataRepository> repo;
    std::unique_ptr<Predictor> predictor;
    httplib::Server server;
};

namespace {

void set_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

void send_error(httplib::Response& res, int status, const std::string& message) {
    set_cors(res);
    res.status = status;
    json::Object out;
    out.field("error", message);
    res.set_content(out.str(), "application/json");
}

std::string query_param(const httplib::Request& req, const std::string& key,
                        const std::string& fallback) {
    auto it = req.params.find(key);
    return it == req.params.end() ? fallback : it->second;
}

int int_param(const httplib::Request& req, const std::string& key, int fallback) {
    const auto raw = query_param(req, key, "");
    if (raw.empty()) return fallback;
    try {
        return std::stoi(raw);
    } catch (...) {
        return fallback;
    }
}

}  // namespace

HttpServer::HttpServer(ServerConfig config,
                       std::unique_ptr<DataRepository> repo,
                       std::unique_ptr<Predictor> predictor)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = std::move(config);
    impl_->repo = std::move(repo);
    impl_->predictor = std::move(predictor);

    auto& server = impl_->server;

    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        res.status = 204;
    });

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        json::Object out;
        out.field("status", "ok");
        res.set_content(out.str(), "application/json");
    });

    server.Get("/history", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto ticker = query_param(req, "ticker", "");
            if (ticker.empty()) { send_error(res, 400, "ticker is required"); return; }
            const int days = int_param(req, "days", impl_->config.history_size);

            auto frame = impl_->repo->load(ticker);
            const auto& closes = frame.column("close");
            const int take = std::min(static_cast<int>(closes.size()), days);
            std::vector<float> history(closes.end() - take, closes.end());
            std::vector<std::string> dates;
            if (static_cast<int>(frame.dates.size()) >= take) {
                dates.assign(frame.dates.end() - take, frame.dates.end());
            }
            json::Object out;
            out.field("ticker", ticker)
               .array("history", history)
               .array("history_dates", dates);
            set_cors(res);
            res.set_content(out.str(), "application/json");
        } catch (const std::exception& e) {
            SV_LOG(Error, "/history failed: " << e.what());
            send_error(res, 500, e.what());
        }
    });

    server.Get("/predict", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto ticker = query_param(req, "ticker", "");
            if (ticker.empty()) { send_error(res, 400, "ticker is required"); return; }
            const auto model = query_param(req, "model", impl_->config.default_model);
            const int days = int_param(req, "days", impl_->config.history_size);

            auto frame = impl_->repo->load(ticker);
            auto features = build_feature_matrix(frame);
            const auto& closes_d = frame.column("close");
            std::vector<float> closes_f(closes_d.begin(), closes_d.end());

            PredictionInput input{ticker, model};
            const auto result = impl_->predictor->predict(input, features, closes_f,
                                                          frame.dates, days);
            json::Object out;
            out.field("ticker", result.ticker)
               .field("model", result.model)
               .field("prediction", static_cast<double>(result.prediction))
               .field("last_close", static_cast<double>(result.last_close))
               .array("history", result.history)
               .array("history_dates", result.history_dates);
            set_cors(res);
            res.set_content(out.str(), "application/json");
        } catch (const std::exception& e) {
            SV_LOG(Error, "/predict failed: " << e.what());
            send_error(res, 500, e.what());
        }
    });

    server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        SV_LOG(Info, req.method << ' ' << req.path << " -> " << res.status);
    });
}

HttpServer::~HttpServer() = default;

void HttpServer::run() {
    SV_LOG(Info, "Listening on " << impl_->config.host << ':' << impl_->config.port);
    if (!impl_->server.listen(impl_->config.host, impl_->config.port)) {
        throw std::runtime_error("Failed to bind to " + impl_->config.host + ":" +
                                 std::to_string(impl_->config.port));
    }
}

}  // namespace stockvision
