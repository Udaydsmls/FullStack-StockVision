#include "features.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stockvision {
namespace {

using Series = std::vector<double>;

Series moving_average(const Series& values, int window) {
    Series out(values.size());
    double running_total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        running_total += values[i];
        if (static_cast<int>(i) >= window) running_total -= values[i - window];
        out[i] = running_total / std::min(window, static_cast<int>(i) + 1);
    }
    return out;
}

Series exponential_average(const Series& values, int span) {
    Series out(values.size());
    if (values.empty()) return out;
    const double weight = 2.0 / (span + 1.0);
    out[0] = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        out[i] = weight * values[i] + (1.0 - weight) * out[i - 1];
    }
    return out;
}

Series relative_strength_index(const Series& close, int period) {
    Series out(close.size(), 50.0);  // 50 is neutral, used until we have two bars
    const double weight = 1.0 / period;
    double average_gain = 0.0;
    double average_loss = 0.0;

    for (std::size_t i = 1; i < close.size(); ++i) {
        const double change = close[i] - close[i - 1];
        const double gain = std::max(change, 0.0);
        const double loss = std::max(-change, 0.0);
        // The first bar seeds the averages; after that they decay exponentially.
        average_gain = i == 1 ? gain : weight * gain + (1.0 - weight) * average_gain;
        average_loss = i == 1 ? loss : weight * loss + (1.0 - weight) * average_loss;
        out[i] = average_loss <= 1e-12 ? 100.0
                                       : 100.0 - 100.0 / (1.0 + average_gain / average_loss);
    }
    return out;
}

Series rolling_std(const Series& values, int window) {
    Series out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t first = i + 1 > static_cast<std::size_t>(window) ? i + 1 - window : 0;
        const std::size_t count = i + 1 - first;
        if (count <= 1) continue;  // a single point has no spread

        double mean = 0.0;
        for (std::size_t j = first; j <= i; ++j) mean += values[j];
        mean /= count;

        double variance = 0.0;
        for (std::size_t j = first; j <= i; ++j) variance += (values[j] - mean) * (values[j] - mean);
        out[i] = std::sqrt(variance / (count - 1));
    }
    return out;
}

}  // namespace

FeatureMatrix build_features(const PriceTable& prices) {
    const std::size_t days = prices.size();
    if (days == 0) throw std::runtime_error("No price rows to build features from");

    const Series& close = prices.close;

    Series log_return(days);
    for (std::size_t i = 1; i < days; ++i) {
        if (close[i - 1] > 0.0) log_return[i] = std::log(close[i] / close[i - 1]);
    }

    const Series sma_10 = moving_average(close, 10);
    const Series sma_30 = moving_average(close, 30);
    const Series ema_12 = exponential_average(close, 12);
    const Series ema_26 = exponential_average(close, 26);
    const Series rsi_14 = relative_strength_index(close, 14);

    Series macd(days);
    for (std::size_t i = 0; i < days; ++i) macd[i] = ema_12[i] - ema_26[i];
    const Series macd_signal = exponential_average(macd, 9);

    const Series middle_band = moving_average(close, 20);
    const Series spread = rolling_std(close, 20);

    // The order here has to match FEATURE_NAMES in params.txt.
    FeatureMatrix matrix;
    matrix.names = {"close", "open", "high", "low", "volume", "log_return",
                    "sma_10", "sma_30", "ema_12", "ema_26", "rsi_14",
                    "macd", "macd_signal", "macd_hist",
                    "bb_upper", "bb_lower", "bb_width"};
    matrix.rows.resize(days);

    for (std::size_t i = 0; i < days; ++i) {
        const double bb_upper = middle_band[i] + 2.0 * spread[i];
        const double bb_lower = middle_band[i] - 2.0 * spread[i];
        matrix.rows[i] = {
            static_cast<float>(close[i]),
            static_cast<float>(prices.open[i]),
            static_cast<float>(prices.high[i]),
            static_cast<float>(prices.low[i]),
            static_cast<float>(prices.volume[i]),
            static_cast<float>(log_return[i]),
            static_cast<float>(sma_10[i]),
            static_cast<float>(sma_30[i]),
            static_cast<float>(ema_12[i]),
            static_cast<float>(ema_26[i]),
            static_cast<float>(rsi_14[i]),
            static_cast<float>(macd[i]),
            static_cast<float>(macd_signal[i]),
            static_cast<float>(macd[i] - macd_signal[i]),
            static_cast<float>(bb_upper),
            static_cast<float>(bb_lower),
            static_cast<float>(bb_upper - bb_lower),
        };
    }
    return matrix;
}

}  // namespace stockvision
