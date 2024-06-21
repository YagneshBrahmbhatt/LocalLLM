from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # Model name remains the same

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_conversation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

prompt = "Alice: How have you been lately?\nBob:"
conversation = generate_conversation(prompt)
print(conversation)
