import React, { useEffect, useState } from "react";

import ErrorBanner from "./components/ErrorBanner";
import Header from "./components/Header";
import PredictionPanel from "./components/PredictionPanel";
import PriceChart from "./components/PriceChart";
import TickerForm from "./components/TickerForm";
import { useModels } from "./hooks/useModels";
import { usePrediction } from "./hooks/usePrediction";

const DEFAULT_TICKER = process.env.REACT_APP_DEFAULT_TICKER || "AAPL";
const DEFAULT_MODEL = process.env.REACT_APP_DEFAULT_MODEL || "lstm";

export default function App() {
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const { models, healthy } = useModels();
  const { data, error, loading, fetchPrediction, reset } = usePrediction();

  useEffect(() => {
    if (models.length > 0 && !models.includes(model)) {
      setModel(models[0]);
    }
  }, [models, model]);

  const handleSubmit = () => {
    fetchPrediction(ticker, model);
  };

  return (
    <div className="min-h-screen px-4 py-10 sm:px-6">
      <main className="mx-auto flex w-full max-w-4xl flex-col gap-6">
        <Header healthy={healthy} />
        <TickerForm
          ticker={ticker}
          onTickerChange={setTicker}
          model={model}
          onModelChange={setModel}
          models={models}
          onSubmit={handleSubmit}
          loading={loading}
        />
        <ErrorBanner message={error} onDismiss={reset} />
        <PredictionPanel data={data} />
        <PriceChart
          history={data?.history}
          historyDates={data?.history_dates}
          prediction={data?.prediction ?? null}
        />
      </main>
    </div>
  );
}
