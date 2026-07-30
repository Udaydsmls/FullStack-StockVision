# StockVision frontend

React (Create React App) + Tailwind UI. The backend row at the top of the page
switches between the FastAPI, C++ and Triton backends at runtime — they all
answer the same REST contract, so only the base URL changes.

## Local development

```bash
cd frontend
cp .env.example .env
npm install
npm start
```

`.env` holds one URL per backend:

```
REACT_APP_FASTAPI_URL=http://localhost:8000
REACT_APP_CPP_URL=http://localhost:8080
REACT_APP_TRITON_URL=http://localhost:8000
```

## Layout

```
frontend/src
├── App.jsx                    # state and data fetching
├── components/
│   ├── Header.jsx             # title + backend health dot
│   ├── Switcher.jsx           # backend picker and forecast/explain tabs
│   ├── TickerForm.jsx         # ticker input, model dropdown, submit
│   ├── PredictionPanel.jsx    # last close, prediction, implied move
│   ├── PriceChart.jsx         # history line + forecast point
│   ├── ExplainChart.jsx       # SHAP feature bars
│   └── ErrorBanner.jsx
├── services/api.js            # backend list + fetch helpers
└── utils/format.js            # currency / percentage helpers
```

The model dropdown is filled from `GET /health`, so an architecture added on
the backend shows up here without a frontend change. The C++ server and Triton
only serve the seven ONNX architectures; Prophet and AutoARIMA are FastAPI
only, as are SHAP explanations.
