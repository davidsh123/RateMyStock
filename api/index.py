"""
RateMyStock — Vercel entrypoint.

Vercel's Python runtime auto-detects Flask by finding an `app` instance at
api/index.py and turns this whole file into a single Vercel Function.
See: https://vercel.com/docs/frameworks/backend/flask

Two routes are exposed:
  GET  /             -> the single-page frontend (templates/index.html)
  GET  /api/analyze  -> JSON sentiment analysis for a ticker
  GET  /api/health   -> simple liveness check
"""
import os
import pickle
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import yfinance

try:
    from google import genai
except ImportError:  # keeps the app importable even if the package is missing
    genai = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "sentiment_model.pkl"
GEMINI_MODEL = "gemini-3.5-flash-lite"
HEADLINE_LIMIT = 5

app = Flask(__name__)

_model = None  # lazily loaded, cached across warm invocations


def get_model():
    """Load the pre-trained Naive Bayes pipeline once per warm container."""
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def naive_bayes_probabilities(model, sentence: str) -> dict:
    proba = model.predict_proba([sentence])[0]
    return {
        "bearish": round(float(proba[0]) * 100, 2),
        "bullish": round(float(proba[1]) * 100, 2),
    }


def pull_headlines(ticker: str, limit: int = HEADLINE_LIMIT) -> list:
    stock = yfinance.Ticker(ticker)
    headlines = []
    for article in (stock.news or [])[:limit]:
        title = (article.get("content") or {}).get("title")
        if title:
            headlines.append(title)
    return headlines


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not genai or not api_key:
        return None
    return genai.Client(api_key=api_key)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze")
def analyze():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Missing 'ticker' query parameter"}), 400

    try:
        headlines = pull_headlines(ticker)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch news for '{ticker}': {exc}"}), 502

    if not headlines:
        return jsonify({"error": f"No recent news found for '{ticker}'"}), 404

    model = get_model()
    client = get_gemini_client()
    chat = client.chats.create(model=GEMINI_MODEL) if client else None

    results = []
    for headline in headlines:
        entry = {
            "headline": headline,
            "naive_bayes": naive_bayes_probabilities(model, headline),
        }
        if chat:
            try:
                entry["gemini"] = chat.send_message(
                    f'Headline: "{headline}". In one short sentence, say whether this '
                    f"is bullish or bearish for {ticker} and why. Plain text only, no "
                    "markdown, no asterisks, no bullet points — a single plain sentence.",
                    config={"tools": [{"url_context": {}}]},
                ).text
            except Exception as exc:
                entry["gemini"] = f"(Gemini unavailable: {exc})"
        else:
            entry["gemini"] = "(Set GEMINI_API_KEY in your Vercel project to enable Gemini analysis)"
        results.append(entry)

    overall = None
    if chat:
        try:
            overall = chat.send_message(
                "Based on these headlines, say in one plain sentence whether this "
                "stock looks like a buy, sell, or neutral. No markdown, no asterisks."
            ).text
        except Exception as exc:
            overall = f"(Gemini unavailable: {exc})"

    return jsonify({"ticker": ticker, "results": results, "overall": overall})


# Local dev: `python api/index.py` (Vercel itself calls the `app` object directly)
if __name__ == "__main__":
    app.run(debug=True)
