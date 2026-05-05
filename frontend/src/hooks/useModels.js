import { useEffect, useState } from "react";

import { api } from "../services/api";

const FALLBACK_MODELS = ["lstm", "bilstm", "gru", "cnn_lstm", "transformer", "tcn", "linear"];

export function useModels() {
  const [models, setModels] = useState(FALLBACK_MODELS);
  const [healthy, setHealthy] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((res) => {
        if (cancelled) return;
        if (Array.isArray(res?.models) && res.models.length > 0) {
          setModels(res.models);
        }
        setHealthy(true);
      })
      .catch(() => {
        if (!cancelled) setHealthy(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { models, healthy };
}
