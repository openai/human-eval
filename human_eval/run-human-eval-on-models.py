# vertex ai
import vertexai
from human_eval.data import write_jsonl, read_problems

import importlib
import argparse
import logging
importlib.invalidate_caches() 
logging.basicConfig(level=logging.INFO)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_id', type=str, default="spherical-frame-413020")
  parser.add_argument('--location', type=str, default="us-west4")
  parser.add_argument('--model', type=str, default="models.gemini_pro")
  return parser.parse_args()
  
def generate_samples(model):
  
  problems = read_problems()
  num_samples_per_task = 1
  #problems = dict(islice(problems.items(), 5))

  samples = [
    dict(task_id=task_id, completion=model.get_response(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
  ]

  write_jsonl("samples2.jsonl", samples)
  
def get_model(model_file_name: str):
  try:
    llm_model = importlib.import_module(model_file_name)
    logging.info('finished loading module {0}'.format(model_file_name))
    return llm_model
  except Exception as e:
    logging.error("Could not load model from config file '{0}' : {1}, skipping".format(model_file_name, str(e)))

if __name__ == "__main__":
  args = parse_args()
  # Initialize Vertex AI
  vertexai.init(project=args.project_id, location=args.location)
  model = get_model(args.model)
  #generate_samples(model)