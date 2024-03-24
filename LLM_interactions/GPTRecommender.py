from openai import OpenAI
from RecommendationTemplateConstructor import RecommendationTemplateConstructor
import json


class GPTRecommender:
    def __init__(self, template_constructor: RecommendationTemplateConstructor) -> None:
        self._client = OpenAI()
        self._template_constructor = template_constructor

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
