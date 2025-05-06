export CUDA_VISIBLE_DEVICES=0,1,2,3

TONGSEARCH_REASONER_PATH=''
BRIGHT_PATH=''
REASONER_FILE_NAME=''
#replace the settings above with your own settings

python run_reasoner_offline.py $TONGSEARCH_REASONER_PATH $BRIGHT_PATH $REASONER_FILE_NAME
sh run_bm25_with_reasoner.sh $REASONED_FILE_NAME-reasoned_query.json
python get_full_results.py 

