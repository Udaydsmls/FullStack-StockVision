import React, { useEffect, useState } from "react";

import ErrorBanner from "./components/ErrorBanner";
import ExplainChart from "./components/ExplainChart";
import Header from "./components/Header";
import PredictionPanel from "./components/PredictionPanel";
import PriceChart from "./components/PriceChart";
import Switcher from "./components/Switcher";
import TickerForm from "./components/TickerForm";
import { api, BACKENDS } from "./services/api";

const TABS = [
  { id: "forecast", label: "Forecast" },
  { id: "explain", label: "Explain" },
];

// Used until a backend tells us what it has; the C++ server and Triton only
// serve the seven architectures that export to ONNX.
const ONNX_MODELS = ["lstm", "bilstm", "gru", "cnn_lstm", "transformer", "tcn", "linear"];

const DEFAULT_TICKER = process.env.REACT_APP_DEFAULT_TICKER || "AAPL";
const DEFAULT_MODEL = process.env.REACT_APP_DEFAULT_MODEL || "lstm";

export default function App() {
  const [backendId, setBackendId] = useState(BACKENDS[0].id);
  const [tab, setTab] = useState("forecast");
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [model, setModel] = useState(DEFAULT_MODEL);

  const [models, setModels] = useState(ONNX_MODELS);
  const [online, setOnline] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Ask whichever backend is selected whether it is up and what it can serve.
  useEffect(() => {
    setOnline(null);
    api
      .health(backendId)
      .then((status) => {
        setOnline(true);
        setModels(status.models?.length ? status.models : ONNX_MODELS);
      })
      .catch(() => setOnline(false));
  }, [backendId]);

  // Fall back to the first available model if the current one is not offered.
  useEffect(() => {
    if (!models.includes(model)) setModel(models[0]);
  }, [models, model]);

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    try {
      if (tab === "forecast") {
        setForecast(await api.predict(backendId, ticker, model));
      } else {
        setExplanation(await api.explain(ticker, model));
      }
    } catch (requestError) {
      setError(requestError.message);
      setForecast(null);
      setExplanation(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen px-4 py-10 sm:px-6">
      <main className="mx-auto flex w-full max-w-4xl flex-col gap-6">
        <Header online={online} />
        <Switcher label="Backend" options={BACKENDS} active={backendId} onChange={setBackendId} />
        <Switcher label="View" options={TABS} active={tab} onChange={setTab} />
        <TickerForm
          ticker={ticker}
          onTickerChange={setTicker}
          model={model}
          onModelChange={setModel}
          models={models}
          onSubmit={handleSubmit}
          loading={loading}
        />
        <ErrorBanner message={error} onDismiss={() => setError(null)} />
        {tab === "forecast" ? (
          <>
            <PredictionPanel data={forecast} />
            <PriceChart
              history={forecast?.history}
              historyDates={forecast?.history_dates}
              prediction={forecast?.prediction}
            />
          </>
        ) : (
          <ExplainChart data={explanation} />
        )}
      </main>
    </div>
  );
}
