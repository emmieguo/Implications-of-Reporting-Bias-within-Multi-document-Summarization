import json
import random
from transformers import BartTokenizer
from datasets import Dataset

# Define the tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Function to tokenize and preprocess the data
def preprocess(examples):
    tokenized = tokenizer(examples["document"], max_length=1024, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=256, truncation=True, padding='max_length')
    return {
        "input_ids": tokenized["input_ids"], 
        "attention_mask": tokenized["attention_mask"], 
        "labels": labels["input_ids"]
    }

# Load JSON data
json_file_path = 'path/to/your/jsonfile.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Shuffle the data to ensure randomness
random.shuffle(data)

# Split the data: 70% train, 15% dev, 15% test
train_split = int(0.7 * len(data))
dev_split = train_split + int(0.15 * len(data))

train_data = data[:train_split]
dev_data = data[train_split:dev_split]
test_data = data[dev_split:]

# Convert lists to Hugging Face Dataset objects
train_dataset = Dataset.from_dict(train_data)
dev_dataset = Dataset.from_dict(dev_data)
test_dataset = Dataset.from_dict(test_data)

# Apply the tokenization
train_dataset = train_dataset.map(preprocess, batched=True)
dev_dataset = dev_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)