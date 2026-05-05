import { useCallback, useState } from "react";

import { api } from "../services/api";

const DEFAULT_DAYS = Number(process.env.REACT_APP_HISTORY_DAYS || 60);

export function usePrediction() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchPrediction = useCallback(async (ticker, model, days = DEFAULT_DAYS) => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.predict(ticker, model, days);
      setData(result);
    } catch (e) {
      setError(e.message || "Request failed");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, error, loading, fetchPrediction, reset };
}
