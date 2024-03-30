import pandas as pd
import feedparser
from urllib.parse import urlparse

class NewsRetriever:
    """Manages retrieval of news from RSS feeds to than be stored in a vector database
    """
    def __init__(self, rss_feeds:list) -> None:
        self.rss_feeds = rss_feeds
        self._news_df = None
        
    def _prepare_dataframe(self):
        self._news_df = pd.DataFrame(columns=['title', 'link', 'domain', 'published', 'summary'])
            
    # def _save_news(self):
    #     # ensure the old file is overwritten
    #     self._news_df.drop_duplicates(inplace=True, subset=['link'])
    #     self._news_df.to_csv(self.csv_database_name, index=False)
            
        
    def retrieve_news(self):
        self._prepare_dataframe()
        for rss_feed in self.rss_feeds:
            print(f'Retrieving news from {rss_feed}...')
            feed = feedparser.parse(rss_feed)
            if feed.bozo == 0:
                for entry in feed.entries:
                    # get the domain name from the url
                    domain = urlparse(entry.link).netloc
                    self._news_df = pd.concat([self._news_df, pd.DataFrame({'title': [entry.title],
                                                                            'link': [entry.link],
                                                                            'domain': [domain],
                                                                            'published': [entry.published],
                                                                            'summary': [entry.summary]}, index=[0])], ignore_index=True)
            else:
                print(f'Error parsing feed {rss_feed}') # implement logging for the whole application later
        return self._news_df