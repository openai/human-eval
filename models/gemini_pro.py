import logging
import textwrap
from typing import Optional

""" Google vertexai imports """
from vertexai.preview.generative_models import GenerativeModel


class GeminiPro:
  """
  A class to interact with the Gemini Pro model from Vertex AI for content generation.

  Attributes:
      defaults (Dict[str, Any]): Default configuration for the model, including the model name
                                  and generation configuration such as temperature and stop sequences.
      model (GenerativeModel): An instance of GenerativeModel from Vertex AI.
  """

  def __init__(self, **kwargs) -> None:
    """
    Initializes the GeminiPro class with default or provided settings.

    Args:
        **kwargs: Optional keyword arguments that can override default model settings.
    """
    self.defaults = {
        "model_name": "gemini-pro",
        "config": {
            "temperature": 0.6,
            "stop_sequences": ["\ndef ", "\nclass ", "\n\n#", "\n\n\n\n", "\n\n"]
        }
    }

    # Update the defaults with any provided overrides
    self.defaults.update(kwargs)

    self.model = GenerativeModel(self.defaults["model_name"])


  def get_response(self, prompt: str) -> Optional[str]:
    """
    Generates a response from the model based on the provided prompt.

    Args:
        prompt (str): The input prompt for content generation.

    Returns:
        Optional[str]: The generated content as a string if successful, None otherwise.
    """
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
      logging.error("GeminiPro response error {0}".format(str(e)))
    return None
