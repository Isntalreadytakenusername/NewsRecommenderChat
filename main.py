# from RSS_feed_collector.NewsRetriever import NewsRetriever
# from vector_database.NewsVectorStorage import NewsVectorStorage

# from LLM_interactions.GPTRecommender import GPTRecommender
# from LLM_interactions.RecommendationTemplateConstructor import RecommendationTemplateConstructor

# import feedparser
# import pandas as pd


# # List of RSS feed URLs
# feeds_urls = [
#     'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
#     'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
#     'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
#     'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
#     'https://rss.nytimes.com/services/xml/rss/nyt/Science.xml',
#     'https://rss.nytimes.com/services/xml/rss/nyt/Health.xml'
# ]

# # Retrieve news from RSS feeds
# news_retriever = NewsRetriever(feeds_urls)
# news_df = news_retriever.retrieve_news()

# # Store news title+summary as embeddings in a vector database with metadata
# news_vector_storage = NewsVectorStorage(news_dataframe=news_df)
# news_vector_storage.load_news()

from fastapi import FastAPI

from fastapi import FastAPI, HTTPException

from LLM_interactions.GPTRecommender import GPTRecommender
from LLM_interactions.RecommendationTemplateConstructor import RecommendationTemplateConstructor
from RSS_feed_collector.NewsRetriever import NewsRetriever
from vector_database.NewsVectorStorage import NewsVectorStorage
from app_requests.UserClick import UserClick
from app_requests.AppInformationHandler import AppInformationHandler

app = FastAPI()

# create a hello return in the root
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/load_new_news")
def load_new_news():
    # List of RSS feed URLs
    feeds_urls = [
    'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Science.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Health.xml'
    ]

    # Retrieve news from RSS feeds
    news_retriever = NewsRetriever(feeds_urls)
    news_df = news_retriever.retrieve_news()

    # Store news title+summary as embeddings in a vector database with metadata
    news_vector_storage = NewsVectorStorage()
    news_vector_storage.load_news(news_dataframe=news_df)

    return {"status": "success"}

@app.get("/get_recommendations/{user_id}")
def get_recommendations(user_id: str):
    template_constructor = RecommendationTemplateConstructor(last_days_interaction=7)
    recommender = GPTRecommender(template_constructor=template_constructor)
    return recommender.get_recommendations(user_id=user_id)

# here I want to enable the app to send the data about what articles the user clicked on

@app.post("/submit_user_click/")
async def submit_name_date(user_click: UserClick):
    # Assuming you want to print or use the data somehow
    print(f"Received a user click. Name: {user_click.user_id}, Date: {user_click.title}, Date: {user_click.date}, Domain: {user_click.domain}")

    try:
        AppInformationHandler.save_user_click(user_click)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}