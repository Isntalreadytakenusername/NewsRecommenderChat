import pandas as pd
from datetime import datetime, timedelta


class RecommendationTemplateConstructor:
    def __init__(self, last_days_interaction = 7) -> None:
        self._user_interaction_history = None
        self._user_preferences = None
        self._last_days_interaction = last_days_interaction
        self.template = '''Articles that user interacted with in the last {days} days: {articles}
User preferences: {preferences}
---------------------
Provide a list of topics that the user might be interested in based on their interaction history and preferences in form of a JSON.
Example: topics_of_interest: ['US Presidential Elections', 'Cake recipies', 'Bitcoin']
            '''
    
    def _get_interaction_history(self, user_id: str) -> None:
        self._user_interaction_history = pd.read_csv(f'UserInteractionHistory/{user_id}_interactions_history.csv', parse_dates=['date'], dayfirst=True)
        # Filter interactions from the last 7 days
        self._user_interaction_history = self._user_interaction_history[self._user_interaction_history['date'] >= (datetime.now() - timedelta(days=self._last_days_interaction))]
    
    def _get_user_preferences(self, user_id: str) -> None:
        with open(f'UserPreferences/{user_id}.txt', 'r') as file:
            self._user_preferences = file.read()
    
    def construct_getting_topics_prompt(self, user_id: str) -> str:
        self._get_interaction_history(user_id)
        self._get_user_preferences(user_id)
        self.template = self.template.format(days=self._last_days_interaction, articles=self._user_interaction_history['title'].tolist(), preferences=self._user_preferences)
        return self.template