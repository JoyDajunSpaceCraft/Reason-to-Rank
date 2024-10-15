import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
from multi_task_model import CustomLlamaForMultiTask, CustomMistralForMultiTask
import argparse 
import pytrec_eval
from tqdm import tqdm

# Function to get qrel file paths based on input name
def get_qrel_file(name):
    print("begin name", name)
    THE_TOPICS = {
        'dl19': './train/train_data/qrel_files/qrels.dl19-passage.txt',
        'dl20': './train/train_data/qrel_files/qrels.dl20-passage.txt',
        'covid': './train/train_data/qrel_files/qrels.beir-v1.0.0-trec-covid.test.txt',
        'touche': './train/train_data/qrel_files/qrels.beir-v1.0.0-webis-touche2020.test.txt',
        'news': './train/train_data/qrel_files/qrels.beir-v1.0.0-trec-news.test.txt',
        'nfcorpus': './train/train_data/qrel_files/qrels.beir-v1.0.0-nfcorpus.test.txt',
        'dbpedia': './train/train_data/qrel_files/qrels.beir-v1.0.0-dbpedia-entity.test.txt',
        'robust04': './train/train_data/qrel_files/qrels.beir-v1.0.0-robust04.test.txt',
    }
    return THE_TOPICS[name]

# Function to load qrels data
def load_qrels(file_path):
    qrels = {}
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)
    return qrels

# Function to load data from JSONL file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Function to run inference and rerank the documents
def inference_rerank(query, docs, adapter_model, device="cuda:0", tokenizer=AutoTokenizer.from_pretrained("Default Model Path")):
    doc_scores = []
    reasons = []
    adapter_model.eval()
    for i, doc in enumerate(docs):
        # Prepare the individual prompt for each document
        prompt = f"I will provide you with a passage enclosed in [], followed by the original text.\nRank the passage based on its relevance to the search query: {query}\n[Passage {i+1}] {doc}\n"

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(adapter_model.device)
        attention_mask = inputs.attention_mask.to(adapter_model.device)

        # Run the model forward pass
        with torch.no_grad():
            outputs = adapter_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract rank logits (the score for this document)
        rank_logits = outputs.rank_logits
        doc_score = rank_logits[:, -1].item()
        doc_scores.append(doc_score)

        # Extract and decode the reason (optional)
        reason_logits = outputs.reason_logits
        reason = tokenizer.decode(reason_logits[0].argmax(dim=-1), skip_special_tokens=True)
        reasons.append(reason)

    # Sort the document scores to obtain the rerank order
    rerank_order = torch.argsort(torch.tensor(doc_scores), descending=True).cpu().tolist()
    return rerank_order, reasons

# Function to get inference results from the model
def get_inferece_run(run_file, device, tokenizer, adapter_model):
    data = load_data(run_file)
    res = {}
    for sample in tqdm(data):
        query = sample['query']
        query_id = sample["qid"]
        docs = sample['unsorted_docs']
        true_relevance = sample['re_rank_id']
        sorted_docids = sample["sorted_docids"]

        # Map true relevance to document ids
        id2docid = {}
        for i, j in zip(true_relevance, sorted_docids):
            id2docid[i] = j

        rerank_order, reason = inference_rerank(query, docs, adapter_model=adapter_model, device=device, tokenizer=tokenizer)

        new_docid = [id2docid[i] for i in rerank_order]

        # Dynamically calculate scores based on document length
        new_score = list(range(len(new_docid), 0, -1))  # Descending score based on rank
        run = {}

        for i, j in zip(new_docid, new_score):
            run[i] = j
        if isinstance(query_id, int):
            query_id = str(query_id)
        res[query_id] = run
    return res

# Main function to run the evaluation
def main(args):
    device = args.device
    save_name = args.save_name
    adapter_config_path = f"{save_name}/adapter_config.json"
    adapter_model_path = f"{save_name}/adapter_model.bin"
    
    # Load the appropriate model based on the input model ID
    if "mistral" in args.model_id:
        base_model = CustomMistralForMultiTask.from_pretrained(args.model_id, torch_dtype=torch.float16)
    else:
        base_model = CustomLlamaForMultiTask.from_pretrained(args.model_id, torch_dtype=torch.float16)
    
    base_model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the adapter model using PEFT
    peft_config = PeftConfig.from_json_file(adapter_config_path)
    adapter_model = PeftModel(base_model, peft_config)
    state_dict = torch.load(adapter_model_path, map_location=device)
    adapter_model.load_state_dict(state_dict, strict=False)
    adapter_model.to(device)
    print("Finished loading model.")

    # Load qrels and initialize evaluator for NDCG calculation
    qrels = load_qrels(get_qrel_file(args.eval_type))
    ndcg_cut = f'ndcg_cut.{args.ndcg}'
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_cut})

    run_file = f"data/validation/listwise_{args.eval_type}_test_5_pid.jsonl"

    # Get the inference results
    res = get_inferece_run(run_file, device=device, tokenizer=tokenizer, adapter_model=adapter_model)
    results = evaluator.evaluate(res)

    total_ndcg = 0
    res_count = 0
    for query_id, query_measures in results.items():
        if query_measures[ndcg_cut] > 0.5:  # Filter low NDCG results
            total_ndcg += query_measures[ndcg_cut]
            res_count += 1

    if res_count > 0:
        average_ndcg = total_ndcg / res_count
        print(f'Overall NDCG@{args.ndcg}: {average_ndcg:.4f}')
    else:
        print(f'No valid NDCG@{args.ndcg} results.')

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate NDCG scores for a multi-task model.')
    parser.add_argument('--save_name', default="PEFT_rank_llama2_7B", type=str, help="Name of the model save directory.")
    parser.add_argument('--model_id', default="llama2_7B", type=str, help="Model ID for loading the base model.")
    parser.add_argument('--eval_type', default="touche", type=str, help="Evaluation type (e.g., touche, dbpedia, etc.).")
    parser.add_argument('--ndcg', default=5, type=int, choices=[5, 10], help="NDCG cut-off value (5 or 10).")
    parser.add_argument('--device', default="cuda:0", type=str, help="Device to run the model on.")
    args = parser.parse_args()
    main(args)
