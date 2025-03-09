from datasets import load_metric
import numpy as np
import json

# load all files
gen_summaries = '/kaggle/input/chatgpt/out_rt_ChatGPT4.txt'
human_summaries = '/kaggle/input/chatgpt/human_rt_ChatGPT4.txt'
original_documents = '/kaggle/input/chatgpt4json/10_rt.json'

def load_summaries(curr_file):
    with open(curr_file, 'r') as file:
        summaries = file.read().strip().split('\n\n')
    return summaries

def load_documents(curr_file):
    with open(curr_file, 'r') as file:
        data = json.load(file)
    return [item["document"] for item in data]

generated = load_summaries(gen_summaries)
human = load_summaries(human_summaries)
original = load_documents(original_documents)

# load and compute scores
bert_score = load_metric("bertscore")
rouge = load_metric("rouge")

bert_scores = bert_score.compute(predictions=generated, references=human, lang="en")
rouge_scores = rouge.compute(predictions=generated, references=human)

avg_bert_scores = {key: np.mean(value) for key, value in bert_scores.items() if isinstance(value, list)}

def data_statistics(docs, summaries):
    doc_lengths = [len(doc.split()) for doc in docs]
    summary_lengths = [len(summary.split()) for summary in summaries]
    return {
        "avg_doc_length": np.mean(doc_lengths),
        "avg_summary_length": np.mean(summary_lengths),
        "compression_ratio": np.mean(summary_lengths) / np.mean(doc_lengths)
    }

data_stats = data_statistics(original, generated)

print("BERT Scores:", avg_bert_scores)
print("ROUGE Scores:", rouge_scores)
print("Data Statistics:", data_stats)