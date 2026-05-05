# StockVision Frontend

React (CRA) UI for the StockVision API. Works with either the Python or C++
backend.

## Local development

```bash
cd frontend
cp .env.example .env       # adjust REACT_APP_API_URL if needed
npm install
npm start
```

By default the UI calls `http://localhost:8000` (the FastAPI service). Set
`REACT_APP_API_URL=http://localhost:8080` in `.env` to point at the C++ server
instead.

## Layout

```
frontend/src
├── App.jsx                    # Top-level orchestrator
├── components/
│   ├── Header.jsx
│   ├── TickerForm.jsx
│   ├── PredictionPanel.jsx
│   ├── PriceChart.jsx
│   └── ErrorBanner.jsx
├── hooks/
│   ├── usePrediction.js       # Prediction request lifecycle
│   └── useModels.js           # Health check + model discovery
├── services/api.js            # Typed-ish API client
└── utils/format.js            # Currency / percentage helpers
```

The app discovers the available model architectures from `GET /health` so any
new model added on the backend appears automatically in the dropdown.
