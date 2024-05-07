from openai import OpenAI
import json
from vector_database.NewsVectorStorage import NewsVectorStorage
import pandas as pd
import os
import logging
import numpy as np

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

        self.news_vector_storage = NewsVectorStorage()
        fh = logging.FileHandler(self._log_filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_topics(self, user_id: str) -> dict:
        """
        Retrieves topics of interest for a given user.

        Args:
            user_id (str): The ID of the user.

        Returns:
            dict: A dictionary with one key containing the topics if interest like topics_of_interest: ['Topic1', 'Topic'...]
        """
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
    
    def get_candidates(self, user_id: str) -> pd.DataFrame:
        """
        Retrieves a dictionary of news articles that are potential candidates for recommendation based on the user's topics of interest.

        Args:
            user_id (str): The ID of the user.

        Returns:
            pd.DataFrame: A DataFrame containing the potential candidates for recommendation (max 30 nearest)
        """
        topics = self.get_topics(user_id)
        topics = topics["topics_of_interest"]
        
        # handle the case when there are no user preferences yet
        if len(topics) == 0:
            random_articles = self.news_vector_storage.query_random()
            return random_articles
        
        news_df = self.news_vector_storage.query_topics(topics)
        return news_df
    
    def get_random_diversified_candidates(self, recommended_titles: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame containing a combination of recommended titles and randomly selected articles.

        Parameters:
        recommended_titles (pd.DataFrame): A DataFrame containing the recommended titles.

        Returns:
        pd.DataFrame: A DataFrame containing a combination of recommended titles and randomly selected articles.
        """
        random_articles = self.news_vector_storage.query_random(5)
        random_articles["explanations"] = "We thought you might like these articles as well."
        return pd.concat([recommended_titles, random_articles], ignore_index=True)
        
    
    def get_recommended_titles(self, user_id: str, candidates: pd.DataFrame) -> dict:
        """
        Ranking part. Retrieves recommended titles for a given user and candidates DataFrame.

        Args:
            user_id (str): The ID of the user.
            candidates (pd.DataFrame): The DataFrame containing the candidate titles.

        Returns:
            dict: A dictionary containing the recommended titles.

        """
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
        """
        Used after ranking. Prepares the response JSON object with recommended titles and adds random.

        Args:
            recommended_titles (pd.DataFrame): A DataFrame containing the recommended titles.
            explanations (list): A list of explanations for the recommended titles.

        Returns:
            dict: The response JSON object with recommended titles and explanations.
        """
        recommended_titles["explanations"] = explanations
        recommended_titles = self.get_random_diversified_candidates(recommended_titles)
        response = {key: list(value.values()) for key, value in recommended_titles.to_dict().items()}
        return response
    
    def _check_whether_user_is_new(self, user_id: str) -> None:
        """
        Checks whether the user is new by verifying the existence of the user preferences file.
        If the file does not exist, it creates an empty file for the user.

        Args:
            user_id (str): The ID of the user.

        Returns:
            None
        """
        if not os.path.exists(f'LLM_interactions/UserPreferences/{user_id}.txt'):
            with open(f'LLM_interactions/UserPreferences/{user_id}.txt', 'w') as file:
                file.write("")
    
    def get_recommendations(self, user_id: str) -> pd.DataFrame:
        """
        The orchestrator function. Retrieves recommendations for a given user.

        Args:
            user_id (str): The ID of the user.

        Returns:
            dict: A dictionary containing the recommended titles. Represents the response JSON object that we return to the client.
        """
        self._check_whether_user_is_new(user_id)
        candidates = self.get_candidates(user_id)
        recommended_json = self.get_recommended_titles(user_id, candidates)
        recommended_titles = recommended_json["candidates"]
        response = self.prepare_response_json(self._current_candidates[self._current_candidates['title'].isin(recommended_titles)], recommended_json["explanations"])
        self.logger.info(f"Response from GPT for recommendations: {response}")
        return response
    
    def adjust_recommendations(self, user_id: str, request: str) -> dict:
        """Adjust recommendations based on user feedback.
        
        Args:
            user_id (str): The ID of the user.
            request (str): The user's adjustment request.
        
        Returns:
            dict: A dictionary containing the response indicating whether the adjustement was successful.
        """
        
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