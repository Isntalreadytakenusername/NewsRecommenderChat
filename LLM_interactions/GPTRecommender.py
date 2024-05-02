from openai import OpenAI
import json
from vector_database.NewsVectorStorage import NewsVectorStorage
import pandas as pd
import os
import logging

class GPTRecommender:
    """Class for interacting with OpenAI's GPT API for generating recommendations
    Args:
        template_constructor (RecommendationTemplateConstructor): object for constructing prompts for GPT
    """
    def __init__(self, template_constructor) -> None:
        self._client = OpenAI()
        self._template_constructor = template_constructor
        self._current_candidates = None
        self._log_filename = "logs.log"
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(self._log_filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_topics(self, user_id: str) -> dict:
        prompt = self._template_constructor.construct_getting_topics_prompt(
            user_id)
        self.logger.info(f"Prompt to get topics: {prompt}")
        
        response = self._client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        self.logger.info(f"Response from GPT for topics: {response}")
        return json.loads(response.choices[0].message.content)
    
    def get_candidates(self, user_id: str) -> dict:
        topics = self.get_topics(user_id)
        topics = topics["topics_of_interest"]
        news_vector_storage = NewsVectorStorage()
        
        # handle the case when the are no use preferences yet
        if len(topics) == 0:
            # take random articles
            print("No topics of interest yet. Taking random articles.")
            random_articles = news_vector_storage.query_random()
            return random_articles
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
    
    def prepare_response_json(self, recommended_titles: pd.DataFrame, explanations: list) -> dict:
        response = {key: list(value.values()) for key, value in recommended_titles.to_dict().items()}
        response["explanations"] = explanations
        return response
    
    def _check_whether_user_is_new(self, user_id: str) -> None:
        # if user preferences file does not exist, create it
        if not os.path.exists(f'LLM_interactions/UserPreferences/{user_id}.txt'):
            with open(f'LLM_interactions/UserPreferences/{user_id}.txt', 'w') as file:
                file.write("")
    
    def get_recommendations(self, user_id: str) -> pd.DataFrame:
        self._check_whether_user_is_new(user_id)
        candidates = self.get_candidates(user_id)
        recommended_json = self.get_recommended_titles(user_id, candidates)
        recommended_titles = recommended_json["candidates"]
        return self.prepare_response_json(self._current_candidates[self._current_candidates['title'].isin(recommended_titles)], recommended_json["explanations"])
    
    def adjust_recommendations(self, user_id: str, request: str) -> dict:
        prompt = self._template_constructor.construct_recommendation_adjustment_prompt(
            user_id, request)
        self.logger.info(f"Prompt to adjust recommendations: {prompt}")
        response = self._client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        adjusted_recommendation = json.loads(response.choices[0].message.content)
        self.logger.info(f"Adjusted recommendation: {adjusted_recommendation}")
        
        if adjusted_recommendation["preferences"] is not None:
            with open(f'LLM_interactions/UserPreferences/{user_id}.txt', 'w') as file:
                file.write(adjusted_recommendation["preferences"])
        return adjusted_recommendation