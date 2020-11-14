#!/bin/bash

for ((i=0; i<4; i++))
do
    export CUDA_VISIBLE_DEVICES=${i} 
    python main.py --batch_size 96 --id2path id2path_part_${i}.csv --ann_file ../annotation/caption.json --data_root /data/home/v-yixwe/data_new/ &
done
wait