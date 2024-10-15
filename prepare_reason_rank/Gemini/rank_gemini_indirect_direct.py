import copy
from tqdm import tqdm
import time
import json
import re
import google.generativeai as genai

class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        import openai
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion

class GeminiClient:
    def __init__(self, keys):
        import pathlib
        import textwrap
        from tqdm import tqdm
        import google.generativeai as genai
        genai.configure(api_key=keys)
#     def chat(self,model, messages, return_text=True, max_tokens=1000, temperature=0):
        
#         client = genai.GenerativeModel(model)
#         client_chat = client.start_chat()
#         response =client_chat.send_message(messages)
#         # response = client_chat.generate_content(messages)
#         return response.text
    def chat(self, model, messages, return_text=True, max_tokens=1000, temperature=0):
        print("model name", model)
        model = genai.GenerativeModel(model, generation_config={"response_mime_type": "application/json"})
        for item in messages:
            item["parts"] = item.pop("content")
            if item["role"] =="assistant":
                item["role"] = "model"
            elif  item["role"] == "system":
                item["role"] = "user"
        last_message = messages[-1]
        messages = messages[:-1]
        
        # print("last_message[parts]", last_message["parts"])
        chat = model.start_chat(history = messages)
        try:
            response = chat.send_message(last_message["parts"])

            if return_text:
                return response.text
            else:
                return response
        except Exception as e:
            print(e)
            time.sleep(10)
            response = chat.send_message(last_message["parts"])

            if return_text:
                return response.text
            else:
                return response
class ClaudeClient:
    def __init__(self, keys):
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(api_key=keys)

    def chat(self, messages, return_text=True, max_tokens=300, *args, **kwargs):
        system = ' '.join([turn['content'] for turn in messages if turn['role'] == 'system'])
        messages = [turn for turn in messages if turn['role'] != 'system']
        if len(system) == 0:
            system = None
        completion = self.anthropic.beta.messages.create(messages=messages, system=system, max_tokens=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.content[0].text
        return completion

    def text(self, max_tokens=None, return_text=True, *args, **kwargs):
        completion = self.anthropic.beta.messages.create(max_tokens_to_sample=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.completion
        return completion


# class LitellmClient:
#     #  https://github.com/BerriAI/litellm
#     def __init__(self, keys=None):
#         self.api_key = keys

#     def chat(self, return_text=True, *args, **kwargs):
#         from litellm import completion
#         response = completion(api_key=self.api_key, *args, **kwargs)
#         if return_text:
#             response = response.choices[0].message.content
#         return response


def convert_messages_to_prompt(messages):
    #  convert chat message into a single prompt; used for completion model (eg davinci)
    prompt = ''
    for turn in messages:
        if turn['role'] == 'system':
            prompt += f"{turn['content']}\n\n"
        elif turn['role'] == 'user':
            prompt += f"{turn['content']}\n\n"
        else:  # 'assistant'
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks
def get_prefix_prompt(query, num, prompt_type="indirect_direct"):
    if prompt_type == "no_reason":
        return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    if prompt_type =="indirect_direct":
        return [
            {
                'role': 'system',
                'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
            },
            {
                'role': 'user',
                'content': (
                    f"I will provide you with {num} passages, each indicated by a number identifier [].\n"
                    "For each passage, briefly generate your reasoning process as follows:\n"
                    "1. Judge whether the passage is not applicable (not very common), where the query does not meet the premise of the passage.\n"
                    "2. Check if the query contains direct evidence. If so, judge whether the query meets or does not meet the passage.\n"
                    "3. If there is no direct evidence, try to infer from existing evidence and answer one question: If the passage is ranked in this order, is it possible that a good passage will miss such information? If impossible, then you can assume that the passage should not be ranked in that order. Otherwise, it should be ranked in that order.\n"
                    "Then, read every passage one-by-one and find the sentence where the method is the direct answer for the query and output the sentence.\n"
                    "Then, rank the passages based on their relevance to the query.\n"
                    "Give highest priority to passages that provide a clear and direct definition or explanation of the keywords related to the query.\n"
                    "Consider both the detailed information and any relevant background context provided in each passage.\n"
                    "Provide clear and concise reasons for the ranking, highlighting the specific parts of the passages that influenced your decision.\n"
                    "Make sure to ignore any irrelevant information and focus on content directly related to the query.\n"
                    "In addition to direct reasons for ranking each passage, also consider listwise reasons. A listwise reason involves comparing passages with each other to determine their relative importance. Specifically, compare passages to identify which ones provide more comprehensive, relevant, and accurate information in relation to the query. When providing listwise reasons, mention specific comparative insights, such as why one passage might be more relevant than another based on the overall context and detail provided."
                )
            },
            {
                'role': 'assistant',
                'content': 'Okay, please provide the passages.'
            }
        ]
    if prompt_type == "direct":
        return [
        {
            'role': 'system',
            'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        },
        {
            'role': 'user',
            'content': (
                f"I will provide you with {num} passages, each indicated by a number identifier [].\n"
                f"First, extract the main keywords from the query: {query}.\n"
                "\tElement 1. For each passage briefly generate your reasoning process: First, judge whether the passage is not applicable (not very common), where the query does not meet the premise of the passage. Then, check if the query contains direct evidence. If so, judge whether the query meets or does not meet the passage. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the passage is rank this order, is it possible that a good passage will miss such information? If impossible, then you can assume that the passage should not rank that order. Otherwise, there is should rank that order."
                "Then, you read every passage one-by-one and find the sentence where method as the direct answer for the query and output the sentence."
                "Then, rank the passages based on their relevance to the query.\n"
                "Give highest priority to passages that provide a clear and direct definition or explanation of the keywords related to the query.\n"
                "Consider both the detailed information and any relevant background context provided in each passage.\n"
                "Provide clear and concise reasons for the ranking, highlighting the specific parts of the passages that influenced your decision.\n"
                "Make sure to ignore any irrelevant information and focus on content directly related to the query."
            )
        },
        {
            'role': 'assistant',
            'content': 'Okay, please provide the passages.'
        }
    ]



def get_post_prompt(query, num, prompt_type="indirect_direct"):
    if prompt_type == "no_reason":
        return (f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.The output should be in JSON format with the following structure, representing rerank results:\n"
            "{\n"
            "  \"ranked_passages\": [\n"
            "    {\n"
            "      \"identifier\": int,\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n")
    if prompt_type=="indirect_direct":
        return (
            f"Search Query: {query}.\n"
            f"Rank the {num} passages above based on their relevance to the search query and the extracted keywords. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first.\n"
            "The output should be in JSON format with the following structure, representing rerank results:\n"
            "{\n"
            "  \"ranked_passages\": [\n"
            "    {\n"
            "      \"identifier\": int,\n"
            "      \"direct_reason\": str,\n"
            "      \"listwise_reason\": str,\n"
            "      \"direct_answer_sentence\": str\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "For each passage, provide the extracted keywords and detailed reasons for its ranking. Ensure to separate direct reasons and listwise reasons clearly. Mention specific parts of the passage that influenced your decision. Ensure the reasons are clear, concise, and directly related to the query, balancing direct definitions or explanations and relevant background information. Only output the JSON structured format."
        )
    if prompt_type=="direct":
        return (
        f"Search Query: {query}.\n"
        f"Rank the {num} passages above based on their relevance to the search query and the extracted keywords. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first.\n"
        "The output should be in JSON dict format with the following structure, representing rerank results:\n"
        """{\n
            "ranked_passages": [\n
            {\n
              "identifier": int,\n
              "reason": str,\n
              "direct_answer_sentence": str\n
            },\n
            ...\n
          ]\n
        }\n"""
        "For each passage, mentioning specific parts of the passage that influenced your decision. Ensure the reasons are clear, concise, and directly related to the query, balancing direct definitions or explanations and relevant background information. And only output JSON structured output."
    )

# # Example usage
# query = "What are the benefits of a ketogenic diet?"
# num_passages = 5

# prefix_prompt = get_prefix_prompt(query, num_passages)
# post_prompt = get_post_prompt(query, num_passages)

# # print(prefix_prompt)
# # print(post_prompt)


def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-4', prompt_type="indirect_direct"):
    query = item['query']
    # print("rank_start in  create_permutation_instruction",rank_start)
    # print("len(item['hits'])", len(item['hits']))
    num = len(item['hits'][rank_start: rank_end])
    # print("num in create_permutation_instruction", num)
    max_length = 300

    messages = get_prefix_prompt(query, num, prompt_type=prompt_type)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # print("content in  create_permutation_instruction", content)
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        # print("content in the reason", content)
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num,prompt_type=prompt_type)})

    return messages


def run_llm(messages, api_key=None, model_name="gpt-3.5-turbo"):
    if 'gpt' in model_name:
        Client = OpenaiClient
    elif 'claude' in model_name:
        Client = ClaudeClient
    else:
        Client = GeminiClient

    agent = Client(api_key)

    response = agent.chat(model=model_name, messages=messages, temperature=0, return_text=True)
    print("response",response)
    return response



def clean_response(response: str):
    # 找到第一次出现的排名序列
    ranking_pattern = re.findall(r'\[\d+\] >', response)
    new_response = ""
    # 提取这些模式中的数字
    ranking_numbers = [int(num) for pattern in ranking_pattern for num in re.findall(r'\d+', pattern)]
    new_response = " ".join([str(i )for i in ranking_numbers])
    # 单独处理最后一个排名元素
    # 查找文本中的最后一个 [数字] 模式
    last_number_match = re.search(r'\[\d+\](?!.*\[\d+\])', response)
    if last_number_match:
        last_number = last_number_match.group().strip('[]')
        new_response +=  " " + last_number
        # ranking_numbers.append(str(last_number))
    print("new_response",new_response)
    return new_response

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name)  # chan
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def main():
    from pyserini.search import LuceneSearcher
    from pyserini.search import get_topics, get_qrels
    import tempfile

    api_key = None  # Your openai key

    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    topics = get_topics('dl19-passage')
    qrels = get_qrels('dl19-passage')

    rank_results = run_retriever(topics, searcher, qrels, k=100)

    new_results = []
    for item in tqdm(rank_results):
        new_item = permutation_pipeline(item, rank_start=0, rank_end=20, model_name='gpt-3.5-turbo',
                                        api_key=api_key)
        new_results.append(new_item)

    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(new_results, temp_file)
    from trec_eval import EvalFunction

    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', 'dl19-passage', temp_file])


if __name__ == '__main__':
    main()
