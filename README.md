# LocalLLM
Mini Assignment: Local LLM Deployment and Interaction

This repository contains scripts to set up and interact with two local language models (LLMs): GPT-Neo and GPT-2. The provided scripts demonstrate how to generate a short story and a conversation using these models.

## Prerequisites

- Python 3.6 or higher
- Pip (Python package installer)

## Installation

Before running the scripts, you need to install the required Python libraries. You can do this using the following command:

```sh
pip install torch transformers
```

## GPT-Neo Setup

Script: run_llama.py
This script uses the EleutherAI/gpt-neo-2.7B model to generate a short story based on a given prompt.

Contents of run_llama.py
from transformers import AutoTokenizer, AutoModelForCausalLM
```
model_name = "EleutherAI/gpt-neo-2.7B"  # Model name remains the same

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a short story
def generate_short_story(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example prompt
prompt = "Once upon a time in a land far, far away,"
story = generate_short_story(prompt)
print(story)
```
Running the Script
To run the script, use the following command:
```
python run_llama.py
```
## GPT-2 Setup

Script: run_gpt2.py
This script uses the gpt2 model to generate a conversation between two characters based on a given prompt.

Contents of run_gpt2.py
```
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # Model name remains the same

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate a conversation
def generate_conversation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example prompt
prompt = "Alice: How have you been lately?\nBob:"
conversation = generate_conversation(prompt)
print(conversation)
```
Running the Script
To run the script, use the following command:
```
python run_gpt2.py
```
## Repository Structure

  run_llama.py: Script to generate a short story using the GPT-Neo model.
  run_gpt2.py: Script to generate a conversation using the GPT-2 model.
  README.md: Documentation for setting up and running the scripts.

##Notes

Ensure you have a stable internet connection when running the scripts for the first time, as the models and tokenizers need to be downloaded.
The scripts are set to generate text based on simple prompts. You can modify the prompts and other parameters (like max_length and num_return_sequences) to suit your requirements.

## Conclusion

This repository provides a simple and effective way to interact with local language models for generating text. Feel free to explore and modify the scripts to create more complex and interesting outputs.

If you encounter any issues or have suggestions for improvements, please feel free to create an issue or submit a pull request.

Happy coding!
