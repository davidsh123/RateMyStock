import os
from dotenv import load_dotenv
from google import genai
import yfinance
import pandas as pd
from  classifier import naivebayes


def pull_articles(ticker: str) -> list:
    '''
    Pull up to 5 headlines from yfinance's Ticker.news attribute related to the stock ticker
    
    '''
    stock = yfinance.Ticker(ticker)

    result = []
    for article in stock.news[:5]:
        result.append(article["content"]["title"])

    return result



def summarize(ticker: str) -> None:
    """
    Write the sentiment analysis for ticker. 
    1. Using the trained naive bayes model, indicate the probability of being bullish/bearish
    Also indicate whether gemini 3.5 flash-lite thinks the article is bullish or bearish
    solely based on the headline with 1-3 bullet points of reasoning. 
    
    """

    headlines = pull_articles(ticker)


    client = genai.Client(api_key=GEMINI_API_KEY)
    chat = client.chats.create(model="gemini-3.5-flash-lite")
    

    nb = naivebayes.naivebayes()

    for sentence in headlines:
        print(f"""Article 1: {sentence}
                    NAIVE BAYES MODEL RESULT: 
                    {nb.predict(sentence)}
                    \n
                    GEMINI SUMMARY: 
                {
                chat.send_message(
                    f'''Based on the headline: "{sentence}", indicate whether this article is 
                    bullish or bearish for {ticker}. Then write 1-3 bullet points
                    summarize your reasoning. Keep your response succinct''',
                    config={
                        "tools": [{"url_context": {}}]
                    }
                ).text
                
                }
                    """)
        
    
    print(chat.send_message("""Based on these headlines, indicate if 
                                this stock is a buy/sell/neutral
                                in 1-2 sentences""").text)




if __name__ == "__main__":

    load_dotenv()
    
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    running = True

    ticker = "AAPL"


    summarize(ticker)


        

        














