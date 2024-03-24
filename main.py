from RSS_feed_collector.NewsRetriever import NewsRetriever
from vector_database.NewsVectorStorage import NewsVectorStorage

import feedparser
import pandas as pd


# List of RSS feed URLs
feeds_urls = [
    'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Science.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Health.xml'
]

from NewsRetriever import NewsRetriever
news_retriever = NewsRetriever(feeds_urls, "rss_feed_data.csv")
news_retriever.retrieve_news()