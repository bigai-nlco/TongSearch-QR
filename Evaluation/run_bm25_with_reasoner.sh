rm -rf outputs
REASONER_FILE=$1
#stackexchange
python run.py --task biology --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task earth_science --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task economics --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task psychology --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task robotics --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task stackoverflow --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task sustainable_living --model bm25_with_reasoner --reasoner_file $REASONER_FILE

#theorem-based
python run.py --task theoremqa_theorems --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task theoremqa_questions --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task aops --model bm25_with_reasoner --reasoner_file $REASONER_FILE

#coding
python run.py --task pony --model bm25_with_reasoner --reasoner_file $REASONER_FILE
python run.py --task leetcode --model bm25_with_reasoner --reasoner_file $REASONER_FILE