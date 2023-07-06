#!/bin/bash
script_path="./Fengshenbang-LM/fengshen/utils/llama_convert/convert_fs_llama_tp.py"
input_dir="llama13b_fs"
output_dir="llama13b_fs_tp4"
python $script_path --input_dir $input_dir --output_dir $output_dir --model_parallel_size 4