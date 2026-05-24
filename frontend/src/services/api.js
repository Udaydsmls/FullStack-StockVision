const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, params = {}) {
  const url = new URL(path, BASE_URL);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") url.searchParams.append(k, v);
  });

  let response;
  try {
    response = await fetch(url.toString(), { method: "GET" });
  } catch (e) {
    throw new ApiError(`Network error: ${e.message}`, 0);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || body.error || detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new ApiError(detail, response.status);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  history: (ticker, days) => request("/history", { ticker, days }),
  predict: (ticker, model, days) => request("/predict", { ticker, model, days }),
  predictTriton: (ticker, model, days) => request("/predict/triton", { ticker, model, days }),
  explain: (ticker, model) => request("/explain", { ticker, model }),
};

export { ApiError, BASE_URL };
