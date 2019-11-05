#!/bin/bash
python ./runner.py --kfold=5 --epochs=500 --methods eSNN:rprop:500:ndata,chopra:rprop:500:ndata --datasets mnist --onehot True --multigpu False --batchsize 200 --hiddenlayers 128,128,128 --gpu 1 --prefix mnisttesting --n 1 --cvsummary True --doevaluation False --seed 42 --printcv True
