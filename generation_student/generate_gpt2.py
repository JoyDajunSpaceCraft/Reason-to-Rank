from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from rouge_score import rouge_scorer

# 加载预训练的 GPT-2 模型
model = GPT2LMHeadModel.from_pretrained('./gpt2_results')
model.to("cuda")
# tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_model')
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from rouge_score import rouge_scorer
from datasets import load_dataset
# from alignscore import AlignScore
# Function to compute ROUGE scores
def compute_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]

    # Aggregate scores
    aggregated_scores = {}
    for score in scores:
        for key in score:
            if key in aggregated_scores:
                aggregated_scores[key].append(score[key].fmeasure)
            else:
                aggregated_scores[key] = [score[key].fmeasure]

    # Calculate average scores
    for key in aggregated_scores:
        aggregated_scores[key] = sum(aggregated_scores[key]) / len(aggregated_scores[key])

    return aggregated_scores
# def compute_align_scores(predictions, references):
#     scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='/home/yuj49/BioLaySumm2024-evaluation_scripts/models/AlignScore/AlignScore-base.ckpt', evaluation_mode='nli_sp')
#     score = scorer.score(contexts=references,claims=predictions)
#     return sum(score)/len(score)

# Example usage
predictions = ["This document is about climate change because it discusses environmental impacts."]
references = ["The reason this document is ranked high is its detailed discussion on climate change impacts."]
rouge_scores = compute_rouge_scores(predictions, references)
print("ROUGE Scores:", rouge_scores)
from rouge_score import rouge_scorer
import json
from tqdm import tqdm
train_names = [
    "../data/GPT3.5/msmarco_train_100.jsonl",
    "../data/GPT3.5/msmarco_train_200.jsonl",
    "../data/GPT3.5/msmarco_train_300.jsonl",
    "../data/GPT3.5/msmarco_train_400.jsonl",
    "../data/GPT3.5/msmarco_train_500.jsonl",
    "../data/GPT3.5/msmarco_train_600.jsonl",
    "../data/GPT3.5/msmarco_train_700.jsonl",
    "../data/GPT3.5/msmarco_train_800.jsonl",
    "../data/GPT3.5/msmarco_train_900.jsonl",
    "../data/GPT3.5/msmarco_train_1000.jsonl",
    "../data/GPT3.5/msmarco_train_1100.jsonl"
]

# val_name = "../data/GPT3.5/msmarco_test.jsonl"
predicted_texts = []
validation_reasons = []
for name in train_names:
    with open(name, "r") as f:
      data = []
      idx = 0
      rouge_l = []
      for line in tqdm(f):
        
        item = json.loads(line)
        query = item['query']
        docs = item['unsorted_docs']
        reason = item['reason']
        validation_reasons.append(reason)
        formatted_docs = [f"[{i+1}] {doc[:100]}" for i, doc in enumerate(docs)]
    
        input_text = f"I will provide you with {len(docs)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}. \
        And give me the reason why you rank them that way. Here is the documents: {' '.join(formatted_docs)}"
    
        test_encodings = tokenizer([input_text], return_tensors="pt", padding=True, max_length=1024).to("cuda")
        outputs = model.generate(**test_encodings, max_new_tokens=512, return_dict_in_generate=True, output_scores=True)
        decoded_texts = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in outputs.sequences]
    
        
        for text in decoded_texts:
            predicted_texts.append(text[len(input_text):])
            print(text[len(input_text):])

rouge_scores = compute_rouge_scores(predicted_texts, validation_reasons)
print("Validation ROUGE Scores:", rouge_scores)
with open("../data/GPT2_results/GPT3.5_generated_reasons_train.txt", "w") as f:
    for reason in predicted_texts:
        f.write(reason + "\n")