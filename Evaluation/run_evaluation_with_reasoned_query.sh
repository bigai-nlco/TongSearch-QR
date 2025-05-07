REASONER_FILE=$1
BRIGHT_PATH=$2
sh run_bm25_with_reasoner.sh $REASONER_FILE $BRIGHT_PATH
python get_full_results.py 