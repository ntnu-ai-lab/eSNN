#!/bin/bash
python ./runner.py --kfold=5 --epochs=2000 --methods eSNN:rprop:2000:split:0.15,chopra:rprop:2000:gabel,gabel:rprop:2000:gabel,t3i1:rprop:2000:split,t1i1,t2i1 --datasets iris,use,eco,glass,heart,car,hay,mam,ttt,pim,bal,who,mon,cmc --onehot True --multigpu False --batchsize 1000 --hiddenlayers 13,13 --gpu 0,1 --prefix=2000epochs --n 5 --cvsummary False --printcv False
