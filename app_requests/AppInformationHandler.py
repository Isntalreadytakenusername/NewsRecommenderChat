from app_requests.UserClick import UserClick
import pandas as pd

class AppInformationHandler:
    def __init__(self):
        pass
    
    @staticmethod
    def save_user_click(user_click: UserClick):
        user_history_path = f"LLM_interactions/UserInteractionHistory/{user_click.user_id}_interactions_history.csv"
        try:
            user_history = pd.read_csv(user_history_path)
        except FileNotFoundError:
            user_history = pd.DataFrame(columns=["title", "date", "domain"])
        
        new_interaction = pd.DataFrame([{
            "title": user_click.title,
            "date": user_click.date,
            "domain": user_click.domain
        }])
        
        user_history = pd.concat([user_history, new_interaction], ignore_index=True)
        user_history.to_csv(user_history_path, index=False)