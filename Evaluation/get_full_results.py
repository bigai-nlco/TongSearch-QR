import os
base_path='outputs/'
items=os.listdir(base_path)
result_list=[]
import json
target='bm25_with_reasoner'
for item in items:
    if target not in item:
        continue
    file_full=f"{base_path}{item}/results.json"
    if not os.path.exists(file_full):
        continue
    with open(file_full) as f:
        result=json.load(f)
    val=result['NDCG@10']
    result_list.append([item,val])
for item in result_list:
    print(item[0],item[1])
import numpy as np
print(np.mean([item[1] for item in result_list]))