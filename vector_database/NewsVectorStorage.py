import chromadb
import pandas as pd


class NewsVectorStorage:
    def __init__(self, news_dataframe) -> None:
        self.news_dataframe = news_dataframe
        self._collection_name = "rss_news"
        self._news_vector_db_client = None
        self._collection = None
        self._prepare_db_client_and_collection()

    def _prepare_db_client_and_collection(self):
        self._news_vector_db_client = chromadb.PersistentClient(path="storage")
        try:
            self._collection = self._news_vector_db_client.get_collection(
                name=self._collection_name)
        except Exception:
            self._collection = self._news_vector_db_client.create_collection(
                name=self._collection_name)
    def load_news(self):
        self._collection.upsert(documents=self.news_dataframe.apply(lambda row: str(row['title']) + ' ' + str(row['summary']), axis=1).tolist(),
                                metadatas=self.news_dataframe[['link', 'domain', 'published']].to_dict(
                                    orient='records'),
                                ids=self.news_dataframe.apply(lambda row: str(row['link']), axis=1).tolist())
        
    def query_topics(self, topics: list):
        results = self._collection.query(
            query_texts=topics,
            n_results=5
        )
        return results
        
