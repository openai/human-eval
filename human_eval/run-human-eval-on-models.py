# vertex ai
import vertexai
from human_eval.data import write_jsonl, read_problems
# from models.gemini_pro import *
# object = GeminiPro()
# object.get_response("DSf")
from itertools import islice
import importlib
import argparse
import logging
importlib.invalidate_caches() 
logging.basicConfig(level=logging.INFO)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_id', type=str, default="spherical-frame-413020")
  parser.add_argument('--location', type=str, default="us-west4")
  parser.add_argument('--model', type=str, default="models.gemini_pro.GeminiPro")
  return parser.parse_args()
  
def generate_samples(model):
  
  problems = read_problems()
  num_samples_per_task = 1
  problems = dict(islice(problems.items(), 2))

  samples = [
    dict(task_id=task_id, completion=model.get_response(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
  ]

  write_jsonl("samples29012873.jsonl", samples)
  
def get_model(module_name: str):
  try:
    module_path, class_name = module_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    logging.info('finished loading module {0}'.format(module_name))
    return getattr(module, class_name)
  except (ImportError, AttributeError) as e:
    logging.error("Could not load model from config file '{0}' : {1}, skipping".format(module_name, str(e)))
    raise ImportError(module_name)

if __name__ == "__main__":
  args = parse_args()
  # Initialize Vertex AI
  vertexai.init(project=args.project_id, location=args.location)
  model = get_model(args.model)
  generate_samples(model)