"""
Trains the multinomial Naive Bayes sentiment classifier and pickles the
fitted sklearn Pipeline to model/sentiment_model.pkl so the Vercel
serverless function can load it instantly instead of retraining on
every cold start.

Run this locally whenever Sentiment_Stock_data.csv changes:

    python3 train_model.py
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training" / "Sentiment_Stock_data.csv"
MODEL_PATH = BASE_DIR / "model" / "sentiment_model.pkl"


def main() -> None:
    df = pd.read_csv(DATA_PATH, nrows=4000)
    df = df.dropna(subset=["Sentence", "Sentiment"])
    df = df[["Sentence", "Sentiment"]]

    X = df["Sentence"]
    Y = df["Sentiment"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                ),
            ),
            ("nb", MultinomialNB()),
        ]
    )

    model = pipeline.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=["bearish (0)", "bullish (1)"]))
    print(confusion_matrix(Y_test, y_pred))
    print(f"Accuracy: {model.score(X_test, Y_test):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\nSaved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
