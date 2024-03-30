import chromadb
import pandas as pd


class NewsVectorStorage:
    def __init__(self) -> None:
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
    def load_news(self, news_dataframe):
        # get the list of textual data that we want to store in the vector database
        news_dataframe.drop_duplicates(subset=['link'], inplace=True)
        documents = news_dataframe.apply(lambda row: str(row['title']) + ' ' + str(row['summary']), axis=1).tolist()
        # get the dictionary of metadata that we want to store in the vector database
        metadatas = metadatas=news_dataframe[['link', 'domain', 'published', 'title', 'summary']].to_dict(orient='records')  #  orient='records' instructs Pandas to represent each row in the DataFrame as a separate dictionary within a list
        # we use link as the unique identifier for each document (article)
        ids = news_dataframe.apply(lambda row: str(row['link']), axis=1).tolist()
        self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        
    def _dataframize_query_results(self, results):
        flattened_data = []
        for dist_group, meta_group in zip(results['distances'], results['metadatas']):
            flattened_data.extend(
                {'distance': dist, **meta}
                for dist, meta in zip(dist_group, meta_group)
            )
        df = pd.DataFrame(flattened_data)
        df = df.drop_duplicates(subset=['link'])
        return df
    
    def query_topics(self, topics: list, articles_limit = 30):
        """Retrieves top 30 most recent (or custom number) articles from vector db
        Based on quering by a list of topics that correspond to user interests.

        Args:
            topics (list): list of user interests (e.g. US Presidential Elections)
            topics_limit (int, optional): maximum # of articles to return. Defaults to 30.

        Returns:
            pandas.DataFrame: returns the query results as a pandas DataFrame with distance, link, domain (of the webpage), published (date) columns
        """
        results = self._collection.query(
            query_texts=topics,
            n_results=5
        )
        results_df = self._dataframize_query_results(results)
        results_df.drop_duplicates(subset=['link'], inplace=True)
        # order by distance ascending and limit the number of articles
        return results_df.sort_values(by='distance').head(articles_limit)
        