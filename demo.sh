#! /bin/bash

# for sine dataset
python run_sine.py --hiddens='1000,1000,1000' --num-masks=1 --learning-rate=0.018 --weight-decay=1e-5 --batch-size=256

# for mnist dataset
python run_mnist.py --hiddens='1000,1000,1000' --num-masks=1 --learning-rate=0.00020 --weight-decay=1e-5 --batch-size=128 --epoch=400

# for S&P data
python run_sp.py --hiddens='1000' --num-masks=1 --learning-rate=0.02 --weight-decay=1e-5 --batch-size=8 --epoch=400

# for stock data
python run_stock.py --hiddens='1000' --num-masks=1 --learning-rate=3e-2 --weight-decay=1e-5 --batch-size=8 --epoch=400

# for billiards data
# with all frequencies
python run_ball.py --hiddens='1000,1000,1000' --num-masks=1 --learning-rate=0.0085 --weight-decay=1e-5 --batch-size=128 --epoch=400
# with the first [0, 28] frequencies
python run_ball.py --hiddens='1000,1000,1000' --num-masks=1 --learning-rate=0.0055 --weight-decay=1e-5 --batch-size=128 --epoch=400 --frequency=29

