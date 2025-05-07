rm -rf outputs
REASONER_FILE=$1
#stackexchange
DATASET_PATH=$2
python run.py --task biology --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task earth_science --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task economics --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task psychology --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task robotics --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task stackoverflow --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task sustainable_living --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH

#theorem-based
python run.py --task theoremqa_theorems --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task theoremqa_questions --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task aops --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH

#coding
python run.py --task pony --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH
python run.py --task leetcode --model bm25_with_reasoner --reasoner_file $REASONER_FILE --dataset_path $DATASET_PATH