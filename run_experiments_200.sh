#!/bin/bash
python ./runner.py --kfold=5 --epochs=200 --methods eSNN:rprop:200:split:0.15,chopra:rprop:200:gabel,gabel:rprop:200:gabel,t3i1:rprop:200:split,t1i1,t2i1 --datasets iris,use,eco,glass,heart,car,hay,mam,ttt,pim,bal,who,mon,cmc --onehot True --multigpu False --batchsize 1000 --hiddenlayers 13,13 --gpu 0,1 --prefix=200epochs --n 5 --cvsummary False --printcv False
