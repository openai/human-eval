""" Google vertexai imports """
from vertexai.preview.generative_models import GenerativeModel
""" """
from typing import Iterable, Dict
import textwrap

class GeminiPro:
  
  def __init__(self, **kwargs ) -> None:
    self.defaults = {
      "model_name": "gemini-pro",
      "config" : {
        "temperature": 0.6,
        "stop_sequences": ["\ndef ", "\nclass ", "\n\n#", "\n\n\n\n", "\n\n"] 
      }
    }
    self.model = GenerativeModel(self.defaults["model_name"])
  
  
  def get_response(self, prompt: str):
    # Query the model
    try:
      response = self.model.generate_content(
          [
              textwrap.dedent(prompt)
          ],
          generation_config=self.defaults["config"]
      )
      return response.text
    except ValueError:
      pass
    except Exception as e:
      print("error")
    