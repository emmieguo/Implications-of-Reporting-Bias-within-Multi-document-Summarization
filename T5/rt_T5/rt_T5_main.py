# !pip install numpy transformers datasets torch sentencepiece bert-score rouge-score

import json
import torch
import sentencepiece
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_metric

# load models, tokenizers, datasets
with open('/kaggle/input/rottentomatoes100/rt_test_100.json', 'r') as file:
    data = json.load(file)

tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# helper functions
def preprocess(x):
    tokenized_data = []
    for item in x:
        tokenized = tokenizer(item["document"], max_length = 512, truncation = True, padding = 'max_length', return_tensors = "pt").to(device)
        tokenized_data.append({"input_ids": tokenized.input_ids.squeeze(0), "attention_mask": tokenized.attention_mask.squeeze(0), "summary": item["summary"]})
    return tokenized_data

def generate(tokenized_item):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(tokenized_item["input_ids"].unsqueeze(0), attention_mask = tokenized_item["attention_mask"].unsqueeze(0), max_length = 150, num_beams = 4, length_penalty = 2.0)
    return tokenizer.decode(output_ids[0], skip_special_tokens = True)

tokenized_data = preprocess(data)
generated_summaries = [generate(item) for item in tokenized_data]

# scores!
rouge = load_metric("rouge")
bert_score = load_metric("bertscore")
rouge_scores = rouge.compute(predictions = generated_summaries, references = [item["summary"] for item in tokenized_data])

# Function to calculate data statistics
def data_statistics(docs, summaries):
    document_lengths = [len(doc.split()) for doc in docs]
    summary_lengths = [len(summary.split()) for summary in summaries]
    return {
        "avg_doc_length": np.mean(document_lengths),
        "avg_summary_length": np.mean(summary_lengths),
        "compression_ratio": np.mean(summary_lengths) / np.mean(document_lengths)
    }
stats = data_statistics([item["document"] for item in data], generated_summaries)

bert_scores = bert_score.compute(predictions = generated_summaries, references = [item["summary"] for item in tokenized_data], lang = "en")
bert_score_means = {}
for key in ['precision', 'recall', 'f1']:
    if key in bert_scores:
        scores = bert_scores[key]
        if isinstance(scores, list) and all(isinstance(score, float) for score in scores):
            bert_score_means[key] = np.mean(scores)
        elif isinstance(scores, torch.Tensor):
            bert_score_means[key] = scores.cpu().numpy().mean()


print("ROUGE Scores:", rouge_scores)
print("BERT Scores:", bert_score_means)
print("Data Statistics:", stats)

# print generated summaries
output_file_path = '/kaggle/working/out_T5_rt.txt'
with open(output_file_path, 'w') as file:
    for summary in generated_summaries:
        file.write(summary + "\n\n")