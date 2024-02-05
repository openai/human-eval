""" Google vertexai imports """
from vertexai.language_models import CodeGenerationModel
import textwrap

class CodeGecko:
  
  def __init__(self, **kwargs ) -> None:
    self.defaults = {
      "model_name": "code-gecko@002",
      "config" : {
        "temperature": 0.6,
        "stop_sequences": ["\ndef ", "\nclass ", "\n\n#", "\n\n\n\n", "\n\n"] 
      }
    }
    self.model = CodeGenerationModel.from_pretrained(self.defaults["model_name"])
    

  def get_response(self, prompt: str):
    # Query the model
    try:
      response = self.model.predict(
          prefix=textwrap.dedent(prompt),
          **self.defaults["config"]
      )
      return response.text
    except ValueError:
      pass