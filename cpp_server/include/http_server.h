#pragma once

#include <memory>
#include <string>

#include "config.h"
#include "data_repository.h"
#include "predictor.h"

namespace stockvision {

class HttpServer {
public:
    HttpServer(ServerConfig config, std::unique_ptr<DataRepository> repo, std::unique_ptr<Predictor> predictor);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stockvision
