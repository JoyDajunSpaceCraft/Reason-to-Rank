new_file = "GPT3.5_generated_reasons.txt"
old_file  = "/home/yuj49/DIAYN/data/GPT3.5/msmarco_test.jsonl"
store_file = "msmarco_test_t5.jsonl"
import json
with open(new_file, "r")as f1, open(old_file,"r")as f2, open(store_file, "w") as f3:
    new_reason = []
    new_item = []
    for i in f1.readlines():
        new_reason.append(i)

    data_indx= 0
    for item in f2.readlines():
        data = json.loads(item)
        data["t5_reason"] = new_reason[data_indx]
        json.dump(data, f3)
        f3.write('\n')
        data_indx+=1