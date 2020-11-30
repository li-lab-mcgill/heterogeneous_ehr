#!/bin/bash

python note_to_training.py \
    //home/mcb/users/pnair6/mimic/data/data_files_2Dec/fold5.csv \
    /home/mcb/users/zwen8/data/plos1/process_impact_analysis/mv/max_df_70/multi/clfs/f5 \
    --max_df 0.7 \
    --max_procs 14 \
    --use_vocab /home/mcb/users/zwen8/data/plos1/process_impact_analysis/mv/max_df_70/multi/train_mixehr/ \
    -v \
    --binarized_mv \

# python note_to_training.py \
#     /home/mcb/users/zwen8/data/plos1/process_impact_analysis/mv/train_phy_nur_res_notes.csv \
#     /home/mcb/users/zwen8/data/plos1/process_impact_analysis/mv/max_df_70/multi/train_mixehr/ \
#     --max_df 0.7 \
#     --max_procs 14 \
#     -m \
#     # --use_vocab /home/mcb/users/zwen8/data/plos1/process_impact_analysis/max_df_70/multi/train_mixehr/