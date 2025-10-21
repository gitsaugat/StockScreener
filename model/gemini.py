from google import genai

class GeminiClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client()

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.text