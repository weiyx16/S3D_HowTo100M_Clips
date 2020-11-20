#!/bin/bash

while getopts "a:g:o:" flag ;
do
    case "${flag}" in
        a) datapath=${OPTARG};;
        g) gpunumber=${OPTARG};;
        o) offset=${OPTARG};;
    esac
done

pip install ffmpeg-python --user
for ((i=0+${offset}; i<${gpunumber}+${offset}; i++))
do
    export CUDA_VISIBLE_DEVICES=$((i-offset))
    python main.py --batch_size 96 --id2path ${datapath}/annotations/id2path_part_${i}.csv --ann_file ${datapath}/annotations/caption_${i}.json --data_root ${datapath} &
done
wait