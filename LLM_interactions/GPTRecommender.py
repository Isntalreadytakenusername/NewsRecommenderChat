from openai import OpenAI
import json
from vector_database.NewsVectorStorage import NewsVectorStorage
import pandas as pd

class GPTRecommender:
    """Class for interacting with OpenAI's GPT API for generating recommendations
    Args:
        template_constructor (RecommendationTemplateConstructor): object for constructing prompts for GPT
    """
    def __init__(self, template_constructor) -> None:
        self._client = OpenAI()
        self._template_constructor = template_constructor
        self._current_candidates = None

    def get_topics(self, user_id: str) -> dict:
        prompt = self._template_constructor.construct_getting_topics_prompt(
            user_id)
        response = self._client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
    
    def get_candidates(self, user_id: str) -> dict:
        topics = self.get_topics(user_id)
        topics = topics["topics_of_interest"]
        news_vector_storage = NewsVectorStorage()
        news_df = news_vector_storage.query_topics(topics)
        return news_df
    
    def get_recommended_titles(self, user_id: str, candidates: pd.DataFrame) -> dict:
        self._current_candidates = candidates
        prompt = self._template_constructor.construct_recommendation_prompt(
            user_id, candidates)
        response = self._client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    
    def get_recommendations(self, user_id: str) -> pd.DataFrame:
        candidates = self.get_candidates(user_id)
        recommended_json = self.get_recommended_titles(user_id, candidates)
        recommended_titles = recommended_json["candidates"]
        return self._current_candidates[self._current_candidates['title'].isin(recommended_titles)], recommended_json["explanations"]