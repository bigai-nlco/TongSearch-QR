MODEL=$1
REASONER_FILE=$2
OUTPUT_DIR=$3
python scripts/evaluation/bright_eval.py --task biology --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task earth_science --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task economics --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task psychology --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task robotics --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task stackoverflow --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task sustainable_living --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 

#theorem-based
python scripts/evaluation/bright_eval.py --task theoremqa_theorems --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task theoremqa_questions --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task aops --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 

#coding
python scripts/evaluation/bright_eval.py --task pony --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR 
python scripts/evaluation/bright_eval.py --task leetcode --model $MODEL --reasoner_file $REASONER_FILE --output_dir $OUTPUT_DIR

# output the full results
python scripts/evaluation/print_eval_results.py --base_path $OUTPUT_DIR  --target $MODEL