import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
# from safetensors.torch import load_file
from multi_task_model import CustomLlamaForMultiTask
import argparse 
import pytrec_eval
from tqdm import tqdm

def get_qrel_file(name):
    print("begin name", name)
    THE_TOPICS = {
        'dl19': 'data/qrel_files/qrels.dl19-passage.txt',
        'dl20': 'data/qrel_files/qrels.dl20-passage.txt',
        'covid': 'data/qrel_files/qrels.beir-v1.0.0-trec-covid.test.txt',
        'touche': 'data/qrel_files/qrels.beir-v1.0.0-webis-touche2020.test.txt',
        'news': 'data/qrel_files/qrels.beir-v1.0.0-trec-news.test.txt',
        'nfcorpus': 'data/qrel_files/qrels.beir-v1.0.0-nfcorpus.test.txt',
        'dbpedia': 'data/qrel_files/qrels.beir-v1.0.0-dbpedia-entity.test.txt',
        'robust04': 'data/qrel_files/qrels.beir-v1.0.0-robust04.test.txt',
    }
    
    return THE_TOPICS[name]  # download from pyserini


# 定义函数加载 qrels 数据
def load_qrels(file_path):
    qrels = {}
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)
    return qrels


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    # data = data[:5]
    return data

def inference_rerank(query, docs,adapter_model, device="cuda:0",tokenizer=AutoTokenizer.from_pretrained("/ocean/projects/med230010p/yji3/llama2_7B")):
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
        
        # Assuming the relevant logit is the last token's output (this can vary based on your model setup)
        doc_score = rank_logits[:, -1].item()  # Extracting the score for this document
        doc_scores.append(doc_score)

        # Extract and decode the reason (optional)
        reason_logits = outputs.reason_logits
        reason = tokenizer.decode(reason_logits[0].argmax(dim=-1), skip_special_tokens=True)
        reasons.append(reason)

    # Now sort the document scores to obtain the rerank order
    rerank_order = torch.argsort(torch.tensor(doc_scores), descending=True).cpu().tolist()
    
    return rerank_order, reasons

def get_inferece_run(run_file, device,tokenizer, adapter_model):
    data = load_data(run_file)
    res = {}
    for sample in tqdm(data):
        query = sample['query']
        query_id = sample["qid"]
        docs = sample['unsorted_docs']
        true_relevance = sample['re_rank_id']  # 真实的排序
        sorted_docids = sample["sorted_docids"]
        unsorted_docids = []
       
        id2docid = {}
        for i, j in zip(true_relevance, sorted_docids):
            id2docid[i] = j
        # print(id2docid)
        # 使用 inference_rerank 函数获取重新排序的文档顺序
        rerank_order, reason = inference_rerank(query, docs, adapter_model=adapter_model, device=device,tokenizer=tokenizer, )

        for idx, i in enumerate(rerank_order):
            rerank_order[idx]+=1
        new_docid = [id2docid[i] for i in rerank_order]
 
        new_score = [5,4,3,2,1]
        run = {}    

        for i, j in zip(new_docid, new_score):
            run[i] = j
        if type(query_id) is int:
            query_id = str(query_id)
        res[query_id] = run
    return res


def main(args):
    device = args.device
    save_name = args.save_name
    adapter_config_path = f"{save_name}/adapter_config.json"
    adapter_model_path = f"{save_name}/adapter_model.bin"
    # Load the base model and tokenizer
    # base_model_name =model_id "/home/yuj49/llama3_8B"  # 基础模型的路径或名称
    
    if args.model_id == "llama2_7B":
        base_model_name =  "/ocean/projects/med230010p/yji3/llama2_7B"  
    elif args.model_id == "mistral_7B_v0.1":
        base_model_name =  "/ocean/projects/med230010p/yji3/mistralai/Mistral-7B-Instruct-v0.1"  
    elif args.model_id == "mistral_7B_v0.3":
        base_model_name =  "/ocean/projects/med230010p/yji3/mistralai/Mistral-7B-Instruct-v0.3"  
    else:
        raise ValueError(f"Unknown model: {args.model_id}") 
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 添加填充标记
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    base_model = CustomLlamaForMultiTask.from_pretrained(base_model_name,
                                                     torch_dtype=torch.float16)
    base_model.to(device)  
    # Load the adapter model using PEFT (Parameter-Efficient Fine-Tuning)
    peft_config = PeftConfig.from_json_file(adapter_config_path)
    # print("peft_config",peft_config)
    new_config = PeftConfig()
    for key, value in peft_config.items():
        setattr(new_config, key, value)


    # 加载适配器模型
    adapter_model = PeftModel(base_model, new_config)
    # state_dict = load_file(adapter_model_path)
    state_dict = torch.load(adapter_model_path, map_location=device)
    adapter_model.load_state_dict(state_dict, strict=False)
    adapter_model.to(device)  
    print("Finished")
    
    qrels = load_qrels(get_qrel_file(args.eval_type))

    # 定义评估器
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.5'})
    run_file = f"data/validation/listwise_{args.eval_type}_test_5_pid.jsonl"

    # this res is from the inference model 
    res = get_inferece_run(run_file,device=device, tokenizer=tokenizer, adapter_model=adapter_model)
    results = evaluator.evaluate(res)    
    total_ndcg_5 = 0
    res_count = 0

    for query_id, query_measures in results.items():
        if query_measures['ndcg_cut_5']==0 or query_measures['ndcg_cut_5'] <=0.5:
            continue
        total_ndcg_5 += query_measures['ndcg_cut_5']
        res_count+=1
    average_ndcg_5 = total_ndcg_5 / res_count

    # 打印整体的 NDCG@5
    print(f'Overall NDCG@5: {average_ndcg_5:.4f}')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--save_name', default="PEFT_rank_llama2_7B", type=str)
    parser.add_argument('--model_id', default="llama2_7B")
    parser.add_argument('--eval_type', default="touche", type=str)
    # parser.add_argument('--eval_path', default="data/validation/listwise_msmarco20_test_5_pid.jsonl", type=str)
    parser.add_argument('--device', default="cuda:0", type=str)
    args = parser.parse_args()
    main(args)