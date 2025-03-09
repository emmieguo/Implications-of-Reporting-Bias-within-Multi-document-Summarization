from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_metric
import torch
import numpy as np

tokenizer = T5Tokenizer.from_pretrained('t5-large')
dataset = load_dataset("multi_news")

def preprocess(x):
    tokenized = tokenizer(x["document"], max_length=512, truncation=True, padding='max_length')  # Adjust max_length as needed
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(x["summary"], max_length=150, truncation=True, padding='max_length')  # Adjust max_length as needed
    return {
        "input_ids": tokenized["input_ids"], 
        "attention_mask": tokenized["attention_mask"], 
        "labels": labels["input_ids"]
    }

train_dataset = dataset["train"].map(preprocess, batched=True, load_from_cache_file=False)
dev_dataset = dataset["validation"].map(preprocess, batched=True, load_from_cache_file=False)
test_dataset = dataset["test"].map(preprocess, batched=True, load_from_cache_file=False)



def generate_summary(batch):
    inputs = torch.tensor(batch['input_ids']).to(model.device)
    attention_mask = torch.tensor(batch['attention_mask']).to(model.device)
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=150, num_beams=4, length_penalty=2.0)
    batch['pred_summary'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

# Evaluate on the test set
test_dataset = tokenized_dataset["test"]
results = test_dataset.map(generate_summary, batched=True, batch_size=8)  # Adjust batch_size based on your GPU capabilities

# Load ROUGE and BERT score metrics
rouge = load_metric("rouge")
bert_score = load_metric("bertscore")
    
# Compute ROUGE and BERT scores
rouge_scores = rouge.compute(predictions=results["pred_summary"], references=test_dataset["summary"])
bert_scores = bert_score.compute(predictions=results["pred_summary"], references=test_dataset["summary"], lang="en")

# Data statistics
def data_statistics(dataset):
    original_lengths = [len(doc.split()) for doc in dataset["document"]]
    summary_lengths = [len(summ.split()) for summ in dataset["summary"]]
    return {
        "avg_original_length": np.mean(original_lengths),
        "avg_summary_length": np.mean(summary_lengths),
        "compression_ratio": np.mean(summary_lengths) / np.mean(original_lengths)
    }

stats_original = data_statistics(dataset["test"])
stats_generated = data_statistics(results)

print("ROUGE Scores:", rouge_scores)
print("BERT Scores:", bert_scores)
print("Data Statistics - Original:", stats_original)
print("Data Statistics - Generated:", stats_generated)
