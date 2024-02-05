"""
LLM eval pipeline on vertex AI LLM Models for the Human-Eval dataset
Using evaluation framework from https://github.com/openai/human-eval/
"""

import vertexai
from human_eval.data import write_jsonl, read_problems
from typing import Dict, Any
import importlib
import argparse
import logging
import time
import json
import subprocess


def parse_args() -> argparse.Namespace:
  """
  Parse command-line arguments.

  This function defines and parses command-line arguments required for the script, including
  project ID, location, and model specification. Defaults are provided for each
  option to facilitate easy testing and deployment.

  Returns:
      An argparse.Namespace object containing the parsed arguments.
  """
  # Create an ArgumentParser object for handling command-line arguments
  parser = argparse.ArgumentParser(description="Parse arguments for interfacing with Vertex AI.")
  
  # Add arguments with default values for project ID, location, model and model_config
  parser.add_argument('--project_id', type=str, default="spherical-frame-413020",
                      help="Project ID for the Vertex AI project. Default: 'spherical-frame-413020'")
  parser.add_argument('--location', type=str, default="us-west4",
                      help="Location for the Vertex AI resources. Default: 'us-west4'")
  parser.add_argument('--model', type=str, default="models.gemini_pro.GeminiPro",
                      help="Model identifier for use in predictions. Default: 'models.gemini_pro.GeminiPro', Other Model 'models.code_gecko.CodeGecko'")
  parser.add_argument('--model_config', type=json.loads, default={},
                      help="Model configurations can be overiden using this paramter. Default: empty json {}")
  # Parse and return the arguments
  return parser.parse_args()


def read_dataset() -> Dict:
  """
  Reads a dataset and returns it as a dictionary.

  This function attempts to read problems from a dataset using the `read_problems` function.

  Returns:
      Dict if the dataset is successfully read, otherwise None.
  """
  try:
    # read_problems() returns a dict of task_id to problem info
    problems = read_problems()
    return problems
  except Exception as error:
    # Handle potential errors in reading problems, such as file not found
    logging.error(f"Failed to read problems: {error}")
    return None


def generate_samples(model: Any, num_samples_per_task: int = 1) -> str:
  """
    Generates sample completions for a set of tasks using a specified model.

    This function reads a set of problems, generates completions for each task
    using the provided model, and saves these samples to a JSONL file with a
    timestamped filename.

    Args:
        model: The model to generate responses with. Must have a get_response method.
        num_samples_per_task: The number of samples to generate per task. Defaults to 1.
    Returns:
        The created file name having the generated samples
    """

  problems = read_dataset()
  try:
    # Generate samples for each problem
    samples = [
        dict(task_id=task_id, completion=model.get_response(problems[task_id]["prompt"]))
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    # Construct a filename using the model class name and current timestamp
    file_name = f"{model.__class__.__name__}{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    logging.info(f'Writing samples to {file_name}')
    write_jsonl(file_name, samples)
    return file_name
  except Exception as error:
    logging.error(f"Error during sample generation or file writing: {error}")


def load_model(module_name: str):
  """
    Loads a specified class from a module dynamically.

    Args:
        module_name (str): The fully qualified class name to load, in the format 'module.submodule.ClassName'.

    Returns:
        The loaded class.

    Raises:
        ImportError: If the module or class cannot be found or loaded.
    """
  try:
    # Split the module name from the class name
    module_path, class_name = module_name.rsplit('.', 1)
    # Dynamically import the specified module
    module = importlib.import_module(module_path)
    logging.info('Finished loading module {0}'.format(module_name))
    return getattr(module, class_name)
  except (ImportError, AttributeError) as e:
    logging.error("Could not load module from the model name '{0}' : {1}, skipping".format(module_name, str(e)))
    raise ImportError(module_name)


def run_evaluation(file_path: str) -> None:
  """
  Runs a functional correctness evaluation on a specified file and writes the results to a text file.
  Args:
      file_path (str): Path to the file being evaluated.

  Raises:
      subprocess.CalledProcessError: If the external command fails.
  """
  try:
    results = subprocess.run(
        ["evaluate_functional_correctness", file_path],
        capture_output=True,
        text=True,
        check=True  # Raises CalledProcessError on non-zero exit status
    )
    report_file = f"{file_path}-results.txt"
    
    # Write the command's stdout to a report file
    with open(report_file, 'w') as fp:
        fp.write(results.stdout)
    
    logging.info(f"Results written to {report_file}")
  except subprocess.CalledProcessError as e:
    logging.error(f"Failed to run evaluation for {file_path}: {e}")
    raise


if __name__ == "__main__":
  
  logging.basicConfig(level=logging.INFO)
  
  # Parse Arguments
  args = parse_args()
  
  # Initialize Vertex AI API Wrapper
  vertexai.init(project=args.project_id, location=args.location)
  
  # Load model using model's module name
  model_class = load_model(args.model)
  
  # Create object of model class
  model = model_class()
  
  # Generate file with all the samples
  file_with_samples = generate_samples(model)
  
  # Run the file with generated samples with human-eval evaulation framework
  run_evaluation(file_with_samples)