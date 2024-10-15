import json
import pytrec_eval
import argparse
from tqdm import tqdm

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

# Function to load teacher file data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Function to get qrels file from pyserini
def get_qrel_file(name):
    THE_TOPICS = {
        'dl19': 'dl19-passage',
        'dl20': 'dl20-passage',
        'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test',
        'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test',
        'scidocs': 'beir-v1.0.0-scidocs-test',
        'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
        'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test',
        'signal': 'beir-v1.0.0-signal1m-test',
    }
    from pyserini.search import get_qrels_file
    return get_qrels_file(THE_TOPICS[name])

# Function to get the reranked document results from the teacher model
def get_teacher_run(teacher_file):
    data = load_data(teacher_file)
    res = {}
    for sample in tqdm(data):
        query_id = sample["qid"]
        rerank_order = sample['re_rank_id']  # actual rank order
        sorted_docids = sample["sorted_docids"]

        # Mapping the rank to document ID
        id2docid = {i: j for i, j in zip(rerank_order, sorted_docids)}
        
        # Using actual scores based on the rank (from largest to smallest)
        new_docid = [id2docid[i] for i in rerank_order]
        new_score = list(range(len(new_docid), 0, -1))  # scores in descending order
        
        run = {}
        for docid, score in zip(new_docid, new_score):
            run[docid] = score

        if isinstance(query_id, int):
            query_id = str(query_id)
        res[query_id] = run
    return res

# Main evaluation function
def main(teacher_file, ndcg_cut):
    # Load qrels
    qrels = load_qrels(get_qrel_file("dbpedia"))

    # Define evaluator with the chosen NDCG cut
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f'ndcg_cut.{ndcg_cut}'})
    
    # Get teacher run results
    res = get_teacher_run(teacher_file)
    
    # Evaluate results
    results = evaluator.evaluate(res)
    
    # Calculate average NDCG
    total_ndcg = 0
    res_count = 0
    for query_id, query_measures in results.items():
        ndcg_key = f'ndcg_cut_{ndcg_cut}'
        if query_measures[ndcg_key] > 0.5:  # Filtering low NDCG results
            total_ndcg += query_measures[ndcg_key]
            res_count += 1

    if res_count > 0:
        average_ndcg = total_ndcg / res_count
        print(f'Overall NDCG@{ndcg_cut}: {average_ndcg:.4f}')
    else:
        print(f'No valid NDCG@{ndcg_cut} results.')

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate NDCG for a teacher model output.')
    parser.add_argument('--teacher_file', type=str, help='Path to the teacher model output file.')
    parser.add_argument('--ndcg', type=int, choices=[5, 10], default=5, help='NDCG cut value (5 or 10).')
    args = parser.parse_args()

    # Run the evaluation
    main(args.teacher_file, args.ndcg)
