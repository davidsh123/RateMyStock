# RateMyStock

## classifier
### naivebayes.ipynb / naivebayes.py
Requirements: \
- jupyter
- notebook
- pandas
- sklearn
This file implements a multinomial naive bayes to classify sentences as bullish or bearish
Sentences are vectorized via term frequency - inverse document frequency vectorizer \
This rewards unique words/phrases
We use multinomial naive bayes because we are working with discrete counts/frequency

\
Caveat: Results are possibly overfitted to the training data. 

\
\
Trained using labelled data from https://www.kaggle.com/datasets/avisheksood/stock-news-sentiment-analysismassive-dataset 
\


## main.py
Requirements: \
- google-genai ("pip3 install google-genai")
- python-dotenv ("pip3 install python-dotenv")

Performs sentiment analysis based off news headlines for a given ticker.\
Use the naive bayes model and LLM's intuition to indicate if each article is bearish or bullish\
with gemini's reasoning. 


## Stack
- pandas
- sklearn
- yfinance
- google.genai
