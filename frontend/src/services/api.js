// All three backends answer the same paths with the same JSON, so switching
// between them only means switching the base URL.
export const BACKENDS = [
  {
    id: "fastapi",
    label: "FastAPI",
    url: process.env.REACT_APP_FASTAPI_URL || "http://localhost:8000",
    predictPath: "/predict",
  },
  {
    id: "cpp",
    label: "C++ / ONNX Runtime",
    url: process.env.REACT_APP_CPP_URL || "http://localhost:8080",
    predictPath: "/predict",
  },
  {
    id: "triton",
    label: "Triton",
    // Triton is reached through the FastAPI service, which forwards the tensor.
    url: process.env.REACT_APP_TRITON_URL || "http://localhost:8000",
    predictPath: "/predict/triton",
  },
];

const HISTORY_DAYS = Number(process.env.REACT_APP_HISTORY_DAYS || 60);

export function findBackend(id) {
  return BACKENDS.find((backend) => backend.id === id) || BACKENDS[0];
}

async function get(baseUrl, path, params = {}) {
  const url = new URL(path, baseUrl);
  Object.entries(params).forEach(([key, value]) => url.searchParams.append(key, value));

  let response;
  try {
    response = await fetch(url.toString());
  } catch (error) {
    throw new Error(`Could not reach ${baseUrl}`);
  }

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || body.error || response.statusText);
  }
  return body;
}

export const api = {
  health: (backendId) => get(findBackend(backendId).url, "/health"),

  predict: (backendId, ticker, model, days = HISTORY_DAYS) => {
    const backend = findBackend(backendId);
    return get(backend.url, backend.predictPath, { ticker, model, days });
  },

  // SHAP runs inside the Python service, so explanations always come from FastAPI.
  explain: (ticker, model) => get(findBackend("fastapi").url, "/explain", { ticker, model }),
};
