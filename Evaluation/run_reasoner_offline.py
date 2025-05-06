from datasets import load_dataset
from config import TONGSEARCH_REASONER_PATH,USE_THINK,BRIGHT_DATASET_PATH,REASONED_FILE_NAME
from vllm_model import VllmModel

if __name__=='__main__':
    if USE_THINK:
        think_flag=True
        system_prompt="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    else:
        think_flag=False
        system_prompt=""
    
    model=VllmModel(TONGSEARCH_REASONER_PATH,system_prompt=system_prompt)
    model.think_flag=think_flag
    import os,json
    query_map={}
    filename=f"{REASONED_FILE_NAME}-reasoned_query.json"
    if os.path.exists(filename):
        with open(filename) as f:
            query_map=json.load(f)
    TARGET_LIST=['biology','earth_science','economics','pony','psychology',
                                    'sustainable_living','aops','theoremqa_theorems',
                                    'theoremqa_questions','robotics','stackoverflow','leetcode']
    for target in TARGET_LIST:
        examples = load_dataset(BRIGHT_DATASET_PATH, 'examples',cache_dir='./cache')[target]
        doc_pairs = load_dataset(BRIGHT_DATASET_PATH, 'documents',cache_dir='./cache')[target]
        doc_ids_dict={}
        for dp in doc_pairs:
            doc_id=dp['id']
            doc_content=dp['content']
            doc_ids_dict[doc_id]=doc_content
        from tqdm import tqdm
        for item in tqdm(examples):
            question=item['query']
            if question in query_map:
                continue
            docs=[doc_ids_dict[doc_id] for doc_id in item['gold_ids']]
            prompt = (f'Instructions:\n'
                    f'1. Identify the essential problem.\n'
                    f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
                    f'3. Draft an answer with as many thoughts as you have.\n'
                    f'Query:{question}\n\n')
            query=model.predict(prompt)
            query_map[question]=query
            with open(filename,'w') as f:
                json.dump(query_map,f,ensure_ascii=False)