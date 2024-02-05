import logging
import textwrap
from typing import Optional

""" Google vertexai imports """
from vertexai.language_models import CodeGenerationModel


class CodeGecko:
  """
  A class for interacting with the CodeGenerationModel from Vertex AI for generating code
  based on a given prompt.

  Attributes:
      defaults (dict): Default configuration for the model, including model name and config options.
      model (CodeGenerationModel): An instance of the CodeGenerationModel.
  """
  
  def __init__(self, **kwargs) -> None:
    """
    Initializes the CodeGecko with default or overridden settings.

    Args:
        **kwargs: Optional keyword arguments to override default model settings.
    """
    self.defaults = {
        "model_name": "code-gecko@002",
        "config": {
            "temperature": 0.6,
            "stop_sequences": ["\ndef ", "\nclass ", "\n\n#", "\n\n\n\n", "\n\n"]
        }
    }
    # Update the defaults with any provided overrides
    self.defaults.update(kwargs)

    self.model = CodeGenerationModel.from_pretrained(self.defaults["model_name"])
    

  def get_response(self, prompt: str) -> Optional[str]:
    """
      Generates a response from the model based on the provided prompt.

      Args:
          prompt (str): The input prompt for code generation.

      Returns:
          Optional[str]: The generated code as a string, or None if an error occurs.
      """
    # Query the model
    try:
      response = self.model.predict(
          prefix=textwrap.dedent(prompt),
          **self.defaults["config"]
      )
      return response.text
    except ValueError:
      pass
    except Exception as e:
        logging.error(f"CodeGecko response error: {e}")
    return None
