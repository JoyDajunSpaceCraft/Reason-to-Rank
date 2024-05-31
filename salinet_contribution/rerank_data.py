import torch
import random

from torch.utils.data import Dataset, DataLoader
from os import path, makedirs
from sklearn.model_selection import train_test_split
import re
# first get the formated reason data
def sperate_reason(reason_text):
  pattern = r'\-\s(?:(?:Passage\s)?\[(\d+)\](?:\s?and\s?(?:Passage\s)?\[(\d+)\])?)(.*?)(?=\n\-|\n\n|\Z)'

  # print("reason_text in sperate_reason", reason_text)
  matches = re.findall(pattern, reason_text, re.DOTALL)

  # 初始化一个字典来存储每个文档的reason
  reasons_dict = {}
  if len(matches) < 5:
    # 处理匹配结果
    for match in matches:
        # 提取文档编号和reason描述
        doc_ids = match[:-1]  # 文档编号部分
        doc_reason = match[-1].strip()  # reason描述部分

        # 对每个文档编号进行处理
        for doc_id in doc_ids:
            if doc_id:  # 确保doc_id不为空
                # 为每个文档编号存储或更新reason描述
                if doc_id in reasons_dict:
                    # 如果同一个文档编号对应多个reason，可以选择合并或选择性保留
                    reasons_dict[doc_id] += " " + doc_reason
                else:
                    reasons_dict[doc_id] = doc_reason
  else:
    for match in matches:
      # print("match", match)

      doc_id, _, doc_reason = match
      reasons_dict[doc_id] = doc_reason
          # print(f"Document ID: {doc_id}, Reason: {doc_reason}\n")

  if len(reasons_dict.keys()) <5:
    # print(idx)
    return None
  

  return reasons_dict


def receive_response(data, reason_name="reason"):
    
    responses = [item["re_rank_id"] for item in data]
    
    def remove_duplicate(response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    new_data = []
    unsorted_score = []
    for item, response in zip(data, responses):
        # reasons = item["reason"]
        reasons = item[reason_name]
        reasons_dict = sperate_reason(reasons)
        # print("reasons_dict", reasons_dict)
        passages = item['unsorted_docs']
        
        unsorted_reasoned_response = [] 
        
        if reasons_dict!=None:
            for idx, passage in enumerate(passages):
                unsorted_reasoned_response.append(passage + "reason" +reasons_dict[str(idx+1)] )
        else:
            unsorted_reasoned_response = [""]*5
        
        # response = clean_response(response)
        response = [int(x) - 1 for x in response]
        response = remove_duplicate(response)
        
        original_rank = [tt for tt in range(len(passages))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        
        new_passages = [passages[ii] for ii in response]
        new_reason_passage = [unsorted_reasoned_response[ii] for ii in response]
        unsorted_score = item["scores"]

        
        new_data.append({'query': item['query'],
                         'retrieved_passages': new_passages,
                         'reasoned_passages':new_reason_passage,
                        'unsorted_score':unsorted_score})
    return new_data


class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # query, docs, reasons = self.data[index]
        # return query, docs, reasons
        item = self.data[item]
        query = item['query']
        unsorted_score =item['unsorted_score']
        retrieved_passages = item["retrieved_passages"]
        reasoned_passages=item['reasoned_passages'] 
        return query, unsorted_score, reasoned_passages

class RerankDiffMaskData():
    def __init__(self, data_path="./", seed=42):
        super()
        self.data_path = data_path
        self.data = self.prepare_data()
        self.seed = seed

    def prepare_data(self):
        templates = self.get_template_list()
        male_stereo, female_stereo = self.get_stereo()
        professions = male_stereo + female_stereo
        
        contexts = []
        counter_contexts = []
        labels = []
        for template in templates:
            for profession in professions:
                context = template.format(profession)
                contexts.append(context)
                if profession in male_stereo:
                    counter_contexts.append(template.format(WOMAN))
                    labels.append(1)
                else:
                    counter_contexts.append(template.format(MAN))
                    labels.append(int())
        data = list(zip(contexts, counter_contexts, labels))
        return data
    
    def get_dataloaders(self, batch_size, shuffle=True, val_split=0.1):
        train_data, test_data = train_test_split(self.data, test_size=val_split, random_state=self.seed)
        train_dataset = Dataset(train_data)
        val_dataset = Dataset(test_data)
        return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), 
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        )

    def get_stereo(self):
        male_stereo = []
        female_stereo = []
        with open(path.join(self.data_path, "professions_male_stereo.json"), "r") as f:
            for l in f:
                for j in eval(l):
                    male_stereo.append(j[0])
        with open(path.join(self.data_path, "professions_female_stereo.json"), "r") as f:
            for l in f:
                for j in eval(l):
                    female_stereo.append(j[0])

        return male_stereo, female_stereo


    

    def get_template_list(self, indices=None):
        '''
         Get list of all considered templates by Vig et al. 2020
         A subset from https://arxiv.org/pdf/1807.11714.pdf Lu et al. 2020
        ''' 
        templates = [
            "The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because",
        ]
        if indices:
            subset_templates = [templates[i - 1] for i in indices]
            print("subset of templates:", subset_templates)
            return subset_templates
        return templates

if __name__ == '__main__':
    data = RerankDiffMaskData(seed=1)
    train_dataloader, val_dataloader = data.get_dataloaders(batch_size=1, shuffle=True, val_split=0.00001)  
    m, f = data.get_stereo()
    print(len(m))
    print(len(f))
    print(len(train_dataloader.dataset.data))
    print(len(val_dataloader.dataset.data))
    for batch in train_dataloader:
       X, Xc, y = batch
       print(X[0]+ " he")
       print(X[0]+ " she")
       print(y[0].item())
       break