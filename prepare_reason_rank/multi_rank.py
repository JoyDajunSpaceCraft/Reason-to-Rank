# prepare the reasons from the paper: https://arxiv.org/abs/2304.09542 
import re
from tqdm import tqdm
from rank_gpt_multi_reason import sliding_windows, create_permutation_instruction, run_llm, receive_permutation
import copy


api_key = "Your code"
def clean_reason(permutation):
    ranking_pattern = re.findall(r'\[\d+\] >', permutation)
    new_response = ""
    ranking_numbers = [int(num) for pattern in ranking_pattern for num in re.findall(r'\d+', pattern)]
    new_response = " ".join([str(i )for i in ranking_numbers])
 
    last_number_match = re.search(r'\[\d+\](?!.*\[\d+\])', permutation)
    if last_number_match:
        last_number = last_number_match.group().strip('[]')
        new_response +=  " " + last_number
        # ranking_numbers.append(str(last_number))
    return new_response
# gpt-4
def single_search(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=api_key):
    query = item["query"]
    if "qid" not in item["hits"][0].keys():
        qid = item["query_id"]
    else:
        qid = item["hits"][0]["qid"]
    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, model_name=model_name)
    # (2) Get ChatGPT predicted permutation
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    # (3) Use permutation to re-rank the passage
    reasoning = permutation
    # print(reasoning)
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

# fix with the current sliding window
def file_search(json_file, rank_start=0, rank_end=3,window_size=20, step=10, model_name='gpt-3.5-turbo'):
    return_format = []
    for item in tqdm(json_file):
        res= single_search(item, rank_start=rank_start, rank_end=rank_end, model_name=model_name)
        return_format.append(res)
    return return_format


def sliding_windows(item=None, rank_start=0, rank_end=20, window_size=5, step=5, model_name='gpt-3.5-turbo',
                    api_key=None):
    slide_item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    res = []
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        
        new_item = single_search(slide_item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
        res.append(new_item)
    return res
import json

from tqdm import tqdm
import json
import os
#Load BM25 
bm25_list = {
   "msmarco20":{
       "train":"../data/BM25/bm25_msmarco_train.json",
        "test":"../data/RankGPTBM25/msmarco_20_test.json",
   },
    # "msmarco19":{
    #     "test":"../data/RankGPTBM25/msmarco_19_test.json",},
    #  "trec_covid": {
    #     "test":"../data/RankGPTBM25/trec_covid_test.json"
    #    },
    # "touche":{
    #     "test":"../data/RankGPTBM25/touche_test.json"},
    # "nfcorpus":{
    #     "test":"../data/RankGPTBM25/nfcorpus_test.json"},
}


def file_single_search(json_file, rank_start=0, rank_end=5, model_name='gpt-3.5-turbo', data_name="msmarco",data_type="train"):

    return_format = []
    file_path = f'../data/{model_store}/{data_name}_{data_type}.jsonl'
    if os.path.exists(file_path):
        os.remove(file_path)
    model_store = "GPT3.5" if model_name == 'gpt-3.5-turbo' else "GPT4"
    for item in tqdm(json_file):
        res= sliding_windows(item, rank_start=rank_start, rank_end=rank_end, model_name=model_name)
        return_format.append(res)
        with open(file_path, 'a') as f:
            for res in return_format:
                json.dump(res, f)
                f.write('\n')
    return return_format
    
def file_sliding_search(json_file, rank_start=0, rank_end=20, window_size=5, step=5, model_name='gpt-3.5-turbo',data_name="msmarco", data_type = "train", index=None):
    return_format = []
    model_store = "GPT3.5" if model_name == 'gpt-3.5-turbo' else "GPT4"
    if index:
        file_path = f'{model_store}/{data_name}_{data_type}_{str(index)}.jsonl'
    else:
        file_path = f'{model_store}/{data_name}/listwise_{data_name}_{data_type}_{str(rank_start)}.jsonl'
    
    for item in tqdm(json_file):
        res= sliding_windows(item=item, rank_start=rank_start, rank_end=rank_end, window_size=window_size, step=step, model_name=model_name, api_key=api_key)
        return_format.append(res)
        # print("res from sliding_windows", sliding_windows)
    
    with open(file_path, 'a') as f:
        for res in return_format:
            for r in res:
                json.dump(r, f)
                f.write('\n')
    return return_format


for data_name, data in bm25_list.items():
    if "train" in data.keys():# previouse is using the test
        test_data = data["train"]
        test_data = json.load(open(test_data))
        test_data = test_data # here I change the 
        # print(len(test_data))
        # Note here there will be 20/5 = 4 for each query 
        file_sliding_search(test_data, rank_start=0, rank_end=5, window_size=5, step=5, model_name='gpt-4',data_name=data_name, data_type="train")