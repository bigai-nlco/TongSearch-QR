import os
import json
import argparse
import numpy as np

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate NDCG@10 results from JSON files.")
    parser.add_argument("--base_path", type=str, default="outputs/",
                        help="Base directory containing result subfolders.")
    parser.add_argument("--target", type=str, required=True,
                        help="Target substring to filter result folders (e.g., 'bm25_with_reasoner').")
    return parser.parse_args()

def main():
    args = parse_args()

    # List all items in the base directory
    items = os.listdir(args.base_path)
    result_list = []

    # Iterate over each folder and collect NDCG@10 results
    for item in items:
        if args.target not in item:
            continue

        file_full = os.path.join(args.base_path, item, "results.json")
        if not os.path.exists(file_full):
            continue

        with open(file_full, "r", encoding="utf-8") as f:
            result = json.load(f)

        ndcg10 = result.get("NDCG@10")
        if ndcg10 is not None:
            result_list.append([item, ndcg10])

    # Print individual results
    for item_name, value in result_list:
        print(item_name, value)

    # Compute and print mean NDCG@10
    if result_list:
        mean_value = np.mean([value for _, value in result_list])
        print("Mean NDCG@10:", mean_value)
    else:
        print("No matching results found.")

if __name__ == "__main__":
    main()



# import os
# base_path='outputs/'
# items=os.listdir(base_path)
# result_list=[]
# import json
# target='bm25_with_reasoner'
# for item in items:
#     if target not in item:
#         continue
#     file_full=f"{base_path}{item}/results.json"
#     if not os.path.exists(file_full):
#         continue
#     with open(file_full) as f:
#         result=json.load(f)
#     val=result['NDCG@10']
#     result_list.append([item,val])
# for item in result_list:
#     print(item[0],item[1])
# import numpy as np
# print(np.mean([item[1] for item in result_list]))