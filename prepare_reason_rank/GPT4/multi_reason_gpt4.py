# prepare the reasons from the paper: https://arxiv.org/abs/2304.09542 
import re
from tqdm import tqdm
from rank_gpt4_indirect_direct import create_permutation_instruction
from rank_gpt4_indirect_direct import run_llm
# from rank_claude_indirect_direct import indirect_direct_receive_permutation
import copy
import argparse
# self 
api_key ="Your Openai Key"

def clean_reason(permutation):
    ranking_pattern = re.findall(r'\[\d+\] >', permutation)
    new_response = ""
    ranking_numbers = [int(num) for pattern in ranking_pattern for num in re.findall(r'\d+', pattern)]
    new_response = " ".join([str(i )for i in ranking_numbers])
    last_number_match = re.search(r'\[\d+\](?!.*\[\d+\])', permutation)
    if last_number_match:
        last_number = last_number_match.group().strip('[]')
        new_response +=  " " + last_number
    return new_response


# gpt-4
def single_search(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=api_key, prompt_type="indirect_direct"):
    query = item["query"]
    if len(item["hits"])==0:
        return None
    if "qid" not in item["hits"][0].keys():
        qid = item["query_id"]
    else:
        qid = item["hits"][0]["qid"]
    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, model_name=model_name, prompt_type=prompt_type)
    # (2) Get ChatGPT predicted permutation
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    if permutation==None:
        return None
    # (3) Use permutation to re-rank the passage
    reasoning = permutation
    print(reasoning)
    # print("reasoning in single search", reasoning)
    # json_str = reasoning.replace("'", '"')
    try:
        reason_dict = json.loads(reasoning)
        re_rank_id = [i["identifier"] for i in reason_dict["ranked_passages"]]
    except Exception as e:
        print("error", e)
        re_rank_id = [0]*5
        reason_dict = reasoning
    response_id = []
    # res = res.split()
    if 0 not in re_rank_id:
        item["hits"] = item["hits"][rank_start: rank_end]
        for i in re_rank_id:
            response_id.append(item["hits"][int(i)-1]["docid"])
    unsorted_docs = []
    for j in item["hits"][rank_start: rank_end]:
        unsorted_docs.append(j["content"])

    scores = []
    for k in item["hits"][rank_start: rank_end]:
        scores.append(k["score"])
    return_format = {"query":query,
                    "qid": qid,
                   "sorted_docids":response_id,
                   "re_rank_id":re_rank_id,
                   "unsorted_docs": unsorted_docs,
                   "reason": reason_dict ,
                   "scores": scores}
    return return_format




def sliding_windows(item=None, rank_start=0, rank_end=20, window_size=5, step=5, model_name='gpt-3.5-turbo',
                    api_key=None, prompt_type="indirect_direct"):
    slide_item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    res = []
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        
        new_item = single_search(slide_item, start_pos, end_pos, model_name=model_name, api_key=api_key, prompt_type=prompt_type)
        
        end_pos = end_pos - step
        start_pos = start_pos - step
        if new_item==None:
            continue
        else:
            res.append(new_item)
    return res
import json

from tqdm import tqdm
import json
import os

#Load BM25 
bm25_list = {
   "msmarco20":{
       # "train":"../data/BM25/bm25_msmarco_train.json"
        "test":"../data/RankGPTBM25/msmarco_20_test.json",
   },
    "msmarco19":{
        "test":"../data/RankGPTBM25/msmarco_19_test.json",},
     "trec_covid": {
        "test":"../data/RankGPTBM25/trec_covid_test.json"
       },
    "touche":{
        "test":"../data/RankGPTBM25/touche_test.json"},
    "nfcorpus":{
        "test":"../data/RankGPTBM25/nfcorpus_test.json"},
    "robust04":{
        "test":"../data/RankGPTBM25/robust04_test.json"},
    "trec_news":{
        "test":"../data/RankGPTBM25/trec_news_test.json"},
    "signal1m":{
    "test":"../data/RankGPTBM25/signal1m_test.json"},
    "dbpedia":{
    "test": "../data/RankGPTBM25/dbpedia_test.json"},
    
}


    
def file_sliding_search(json_file, rank_start=0, rank_end=20, window_size=5, step=5, model_name='gpt-3.5-turbo',data_name="msmarco", data_type = "train", index=None, prompt_type="indirect_direct"):
    return_format = []
    if "gpt" in model_name:
        model_store = "GPT3.5" if model_name == 'gpt-3.5-turbo' else "GPT4"
    elif "gemini" in model_name:
        model_store = "Gemini"
    elif "claude" in model_name:
        model_store = "Claude"
    if index:
        file_path = f'{model_store}/{data_name}_{data_type}_{str(index)}.jsonl'
    else:
        if prompt_type =="indirect_direct":
            file_path = f'{model_store}/{data_name}/listwise_{data_name}_{data_type}_{str(rank_end)}.jsonl'
        elif prompt_type =="direct":
            file_path = f'{model_store}/{data_name}/{"direct"}_{data_name}_{data_type}_{str(rank_end)}.jsonl'
        else:
            file_path = f'{model_store}/{data_name}/{"no_reason"}_{data_name}_{data_type}_{str(rank_end)}.jsonl'
    
    
    for item in tqdm(json_file):
        res= sliding_windows(item=item, rank_start=rank_start, rank_end=rank_end, window_size=window_size, step=step, model_name=model_name, api_key=api_key,prompt_type=prompt_type)
        return_format.append(res)
    
    with open(file_path, 'a') as f:
        for res in return_format:
            for r in res:
                json.dump(r, f)
                f.write('\n')
    return return_format

def main(args):
    data_name = args.topic
    data = bm25_list[data_name]
    if "test" in data.keys():# previouse is using the test
        test_data = data["test"]
        test_data = json.load(open(test_data))
        test_data = test_data # here I change the 
        file_sliding_search(test_data, rank_start=0, rank_end=5, window_size=5, step=5, model_name='gpt-4',data_name=data_name, data_type="test", prompt_type=args.prompt_type)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the name of the needed topic")

    parser.add_argument('--topic', type=str, default="msmarco20")
    parser.add_argument('--prompt_type', type=str, default="indirect_direct")
    args = parser.parse_args()
    main(args)
