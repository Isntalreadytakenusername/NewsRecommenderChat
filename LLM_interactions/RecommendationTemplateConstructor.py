import pandas as pd
from datetime import datetime, timedelta


class RecommendationTemplateConstructor:
    def __init__(self, last_days_interaction = 7) -> None:
        self._user_interaction_history = None
        self._user_preferences = None
        self._last_days_interaction = last_days_interaction
        self.template_topics = self.read_template('LLM_interactions/templates/template_topics.txt')
        self.template_recommendations = self.read_template('LLM_interactions/templates/template_recommendation.txt')
        self.template_recommendation_adjustment = self.read_template('LLM_interactions/templates/template_recommendation_adjustment.txt')
    
    def read_template(self, template_path: str) -> str:
        with open(template_path, 'r') as file:
            return file.read()
    
    def _get_interaction_history(self, user_id: str) -> None:
        try:
            self._user_interaction_history = pd.read_csv(f'LLM_interactions/UserInteractionHistory/{user_id}_interactions_history.csv', parse_dates=['date'], dayfirst=True)
            # Filter interactions from the last 7 days
            self._user_interaction_history = self._user_interaction_history[self._user_interaction_history['date'] >= (datetime.now() - timedelta(days=self._last_days_interaction))]
        except FileNotFoundError:
            self._user_interaction_history = pd.DataFrame(columns=["title", "date", "domain"])
    
    def _get_user_preferences(self, user_id: str) -> None:
        try:
            with open(f'LLM_interactions/UserPreferences/{user_id}.txt', 'r') as file:
                self._user_preferences = file.read()
        except FileNotFoundError:
            self._user_preferences = "None"
    
    def construct_getting_topics_prompt(self, user_id: str) -> str:
        self._get_interaction_history(user_id)
        self._get_user_preferences(user_id)
        self.template_topics = self.template_topics.format(days=self._last_days_interaction, articles=self._user_interaction_history['title'].tolist(), preferences=self._user_preferences)
        return self.template_topics
    
    def construct_recommendation_prompt(self, user_id: str, candidates:pd.DataFrame) -> str:
        self._get_interaction_history(user_id)
        self._get_user_preferences(user_id)
        candidate_titles = candidates['title'].tolist()
        self.template_recommendations = self.template_recommendations.format(days=self._last_days_interaction,
                                                                             articles=self._user_interaction_history['title'].tolist(),
                                                                             preferences=self._user_preferences, candidates=candidate_titles)
        return self.template_recommendations
    
    def construct_recommendation_adjustment_prompt(self, user_id: str, request:str) -> str:
        self._get_user_preferences(user_id)
        self.template_recommendation_adjustment = self.template_recommendation_adjustment.format(preferences=self._user_preferences, request=request)
        return self.template_recommendation_adjustment
    