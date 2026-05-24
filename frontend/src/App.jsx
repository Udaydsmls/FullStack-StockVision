import React, { useEffect, useState } from "react";

import ErrorBanner from "./components/ErrorBanner";
import ExplainChart from "./components/ExplainChart";
import Header from "./components/Header";
import PredictionPanel from "./components/PredictionPanel";
import PriceChart from "./components/PriceChart";
import TabSwitcher from "./components/TabSwitcher";
import TickerForm from "./components/TickerForm";
import { useExplanation } from "./hooks/useExplanation";
import { useModels } from "./hooks/useModels";
import { usePrediction } from "./hooks/usePrediction";

const TABS = [
  { id: "forecast", label: "Forecast" },
  { id: "explain", label: "Explain" },
];

const DEFAULT_TICKER = process.env.REACT_APP_DEFAULT_TICKER || "AAPL";
const DEFAULT_MODEL = process.env.REACT_APP_DEFAULT_MODEL || "lstm";

export default function App() {
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [tab, setTab] = useState("forecast");
  const { models, healthy } = useModels();
  const { data, error, loading, fetchPrediction, reset } = usePrediction();
  const explanation = useExplanation();

  useEffect(() => {
    if (models.length > 0 && !models.includes(model)) {
      setModel(models[0]);
    }
  }, [models, model]);

  const handleSubmit = () => {
    if (tab === "forecast") {
      fetchPrediction(ticker, model);
    } else {
      explanation.fetchExplanation(ticker, model);
    }
  };

  const activeError = tab === "forecast" ? error : explanation.error;
  const activeReset = tab === "forecast" ? reset : explanation.reset;
  const activeLoading = tab === "forecast" ? loading : explanation.loading;

  return (
    <div className="min-h-screen px-4 py-10 sm:px-6">
      <main className="mx-auto flex w-full max-w-4xl flex-col gap-6">
        <Header healthy={healthy} />
        <TabSwitcher tabs={TABS} active={tab} onChange={setTab} />
        <TickerForm
          ticker={ticker}
          onTickerChange={setTicker}
          model={model}
          onModelChange={setModel}
          models={models}
          onSubmit={handleSubmit}
          loading={activeLoading}
        />
        <ErrorBanner message={activeError} onDismiss={activeReset} />
        {tab === "forecast" ? (
          <>
            <PredictionPanel data={data} />
            <PriceChart
              history={data?.history}
              historyDates={data?.history_dates}
              prediction={data?.prediction ?? null}
            />
          </>
        ) : (
          <ExplainChart data={explanation.data} />
        )}
      </main>
    </div>
  );
}
