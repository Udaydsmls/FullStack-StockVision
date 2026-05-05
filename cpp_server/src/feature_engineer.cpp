#include "feature_engineer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stockvision {

namespace {

std::vector<double> sma(const std::vector<double>& s, int window) {
    std::vector<double> out(s.size(), 0.0);
    double rolling = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        rolling += s[i];
        if (static_cast<int>(i) >= window) rolling -= s[i - window];
        const int divisor = std::min(window, static_cast<int>(i) + 1);
        out[i] = rolling / divisor;
    }
    return out;
}

std::vector<double> ema(const std::vector<double>& s, int span) {
    std::vector<double> out(s.size(), 0.0);
    if (s.empty()) return out;
    const double alpha = 2.0 / (span + 1.0);
    out[0] = s[0];
    for (std::size_t i = 1; i < s.size(); ++i) {
        out[i] = alpha * s[i] + (1.0 - alpha) * out[i - 1];
    }
    return out;
}

std::vector<double> rsi(const std::vector<double>& close, int period = 14) {
    std::vector<double> out(close.size(), 50.0);
    if (close.size() < 2) return out;
    const double alpha = 1.0 / period;
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    for (std::size_t i = 1; i < close.size(); ++i) {
        const double delta = close[i] - close[i - 1];
        const double gain = delta > 0 ? delta : 0.0;
        const double loss = delta < 0 ? -delta : 0.0;
        if (i == 1) {
            avg_gain = gain;
            avg_loss = loss;
        } else {
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;
        }
        if (avg_loss <= 1e-12) {
            out[i] = 100.0;
        } else {
            const double rs = avg_gain / avg_loss;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    return out;
}

std::vector<double> rolling_std(const std::vector<double>& s, int window) {
    std::vector<double> out(s.size(), 0.0);
    for (std::size_t i = 0; i < s.size(); ++i) {
        const std::size_t lo = (i + 1 > static_cast<std::size_t>(window)) ? i + 1 - window : 0;
        const std::size_t n = i + 1 - lo;
        if (n <= 1) {
            out[i] = 0.0;
            continue;
        }
        double mean = 0.0;
        for (std::size_t j = lo; j <= i; ++j) mean += s[j];
        mean /= n;
        double var = 0.0;
        for (std::size_t j = lo; j <= i; ++j) var += (s[j] - mean) * (s[j] - mean);
        out[i] = std::sqrt(var / (n - 1));
    }
    return out;
}

}  // namespace

FeatureMatrix build_feature_matrix(const OhlcvFrame& frame) {
    const auto& close = frame.column("close");
    const auto& open  = frame.column("open");
    const auto& high  = frame.column("high");
    const auto& low   = frame.column("low");
    const auto& volume = frame.column("volume");
    const std::size_t n = close.size();
    if (n == 0) throw std::runtime_error("Empty OHLCV frame");

    std::vector<double> log_return(n, 0.0);
    for (std::size_t i = 1; i < n; ++i) {
        log_return[i] = (close[i - 1] > 0.0) ? std::log(close[i] / close[i - 1]) : 0.0;
    }

    const auto sma10 = sma(close, 10);
    const auto sma30 = sma(close, 30);
    const auto ema12 = ema(close, 12);
    const auto ema26 = ema(close, 26);
    const auto rsi14 = rsi(close, 14);

    std::vector<double> macd_line(n);
    for (std::size_t i = 0; i < n; ++i) macd_line[i] = ema12[i] - ema26[i];
    const auto macd_signal = ema(macd_line, 9);
    std::vector<double> macd_hist(n);
    for (std::size_t i = 0; i < n; ++i) macd_hist[i] = macd_line[i] - macd_signal[i];

    const auto sma20 = sma(close, 20);
    const auto std20 = rolling_std(close, 20);
    std::vector<double> bb_upper(n), bb_lower(n), bb_width(n);
    for (std::size_t i = 0; i < n; ++i) {
        bb_upper[i] = sma20[i] + 2.0 * std20[i];
        bb_lower[i] = sma20[i] - 2.0 * std20[i];
        bb_width[i] = bb_upper[i] - bb_lower[i];
    }

    FeatureMatrix matrix;
    matrix.column_names = {
        "close", "open", "high", "low", "volume",
        "log_return", "sma_10", "sma_30", "ema_12", "ema_26", "rsi_14",
        "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_width",
    };
    matrix.rows.resize(n, std::vector<float>(matrix.column_names.size()));

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t k = 0;
        matrix.rows[i][k++] = static_cast<float>(close[i]);
        matrix.rows[i][k++] = static_cast<float>(open[i]);
        matrix.rows[i][k++] = static_cast<float>(high[i]);
        matrix.rows[i][k++] = static_cast<float>(low[i]);
        matrix.rows[i][k++] = static_cast<float>(volume[i]);
        matrix.rows[i][k++] = static_cast<float>(log_return[i]);
        matrix.rows[i][k++] = static_cast<float>(sma10[i]);
        matrix.rows[i][k++] = static_cast<float>(sma30[i]);
        matrix.rows[i][k++] = static_cast<float>(ema12[i]);
        matrix.rows[i][k++] = static_cast<float>(ema26[i]);
        matrix.rows[i][k++] = static_cast<float>(rsi14[i]);
        matrix.rows[i][k++] = static_cast<float>(macd_line[i]);
        matrix.rows[i][k++] = static_cast<float>(macd_signal[i]);
        matrix.rows[i][k++] = static_cast<float>(macd_hist[i]);
        matrix.rows[i][k++] = static_cast<float>(bb_upper[i]);
        matrix.rows[i][k++] = static_cast<float>(bb_lower[i]);
        matrix.rows[i][k++] = static_cast<float>(bb_width[i]);
    }
    return matrix;
}

}  // namespace stockvision
