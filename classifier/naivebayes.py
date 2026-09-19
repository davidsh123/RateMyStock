import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


class naivebayes:

    model : Pipeline

    def __init__(self) -> None:
        """
        Trains the naive bayes model using sklearn
        """


        df = pd.read_csv("classifier/sentiment_clean.csv")
        df.dropna(subset = ["text", "label"])
        df = df[["text", "label"]]

        X = df["text"]
        Y = df["label"]

        X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                            Y,
                                                            test_size=0.2,      
                                                            random_state=1)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3), # include both single words, bigrams, trigrams (e.g. "net loss narrowed")
            )),
            ("nb", MultinomialNB()), # multinomial because we are working with discrete counts/frequency
        ])


        self.model = pipeline.fit(X_train, Y_train)



    def predict(self, sentence: str) -> int:
        """
        displays the probabilty of a sentence being bearish/neutral or bullish
        
        """


        result = self.model.predict_proba([sentence])[0]

        return f"""Probability of Bearish/Neutral: {(100*result[0]):.2f}%
                   Probability of Bullish: {(100*result[1]):.2f}%"""
    




