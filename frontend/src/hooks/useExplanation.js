import { useCallback, useState } from "react";

import { api } from "../services/api";

export function useExplanation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchExplanation = useCallback(async (ticker, model) => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.explain(ticker, model);
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

  return { data, error, loading, fetchExplanation, reset };
}
