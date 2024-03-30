import pandas as pd
from datetime import datetime, timedelta


class RecommendationTemplateConstructor:
    def __init__(self, last_days_interaction = 7) -> None:
        self._user_interaction_history = None
        self._user_preferences = None
        self._last_days_interaction = last_days_interaction
        self.template_topics = '''Articles that user interacted with in the last {days} days: {articles}
User preferences: {preferences}
---------------------
Provide a list of topics that the user might be interested in based on their interaction history and preferences in form of a JSON.
Make sure the topics you suggest make sense for news recommendations. Do not produce random topics.
Example: topics_of_interest: ['US Presidential Elections', 'Cake recipies', 'Bitcoin']
            '''
        self.template_recommendations = '''Articles that user interacted with in the last {days} days: {articles}
        User preferences: {preferences}
        ---------------------
        Here are the potential articles that the user might be interested in:
        {candidates}
        Can you provide a list of 10 articles that the user might be interested in based on their interaction history and preferences in form of a JSON along with explanations why you have recommended the article?
        Example: cadidates: ['Title1', 'Title2', 'Title3', 'Title4', 'Title5', 'Title6', 'Title7', 'Title8', 'Title9', 'Title10'], explanations: ['Explanation1', 'Explanation2', 'Explanation3', 'Explanation4', 'Explanation5', 'Explanation6', 'Explanation7', 'Explanation8', 'Explanation9', 'Explanation10']
        Ensure you keep the title names as they are in the data provided.'''
    
    def _get_interaction_history(self, user_id: str) -> None:
        self._user_interaction_history = pd.read_csv(f'LLM_interactions/UserInteractionHistory/{user_id}_interactions_history.csv', parse_dates=['date'], dayfirst=True)
        # Filter interactions from the last 7 days
        self._user_interaction_history = self._user_interaction_history[self._user_interaction_history['date'] >= (datetime.now() - timedelta(days=self._last_days_interaction))]
    
    def _get_user_preferences(self, user_id: str) -> None:
        with open(f'LLM_interactions/UserPreferences/{user_id}.txt', 'r') as file:
            self._user_preferences = file.read()
    
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