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

from LLM_interactions.GPTRecommender import GPTRecommender
from LLM_interactions.RecommendationTemplateConstructor import RecommendationTemplateConstructor


app = FastAPI()

@app.get("/get_recommendations/{user_id}")
def get_recommendations(user_id: str):
    template_constructor = RecommendationTemplateConstructor(last_days_interaction=7)
    recommender = GPTRecommender(template_constructor=template_constructor)
    return recommender.get_recommendations(user_id=user_id)


