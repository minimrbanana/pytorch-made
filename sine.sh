#! /bin/bash

python run_sine.py --hiddens='300' --num_masks=1 --learning_rate=1e-4 --weight_decay=1e-5 --batch_size=32
python run_sine.py --hiddens='300' --num_masks=1 --learning_rate=1e-4 --weight_decay=1e-5 --batch_size=64
