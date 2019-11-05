from __future__ import absolute_import, division, print_function
from models.model_utils import makeANNModel
from utils.keras_utils import set_keras_growth
from utils.storage_utils import createdir
from datetime import datetime

from keras.wrappers.scikit_learn import KerasClassifier
from utils.runutils import runalldatasets,getArgs
import sys
import numpy as np
import pandas as pd
import json
import random


def main():
    args = getArgs()

    if args.seed is None:
        seed = random.randrange(sys.maxsize)
        args.seed = seed
        print(f"generating new random seed: {seed}")
    else:
        print(f"setting random seed to: {args.seed}")

    random.seed(args.seed)

    datasetlist = args.datasets

    print(f"doing experiment with {datasetlist} in that order")
    #ds = Datasets(filename="datasetfiledb")
    k = args.kfold
    results = {}
    runlist = args.methods

    #rootpath = str(os.getpid())
    #rootpath = "gabel_hyper-"+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #createdir(rootpath)

    set_keras_growth(args.gpu)

    prefix = "runner"
    if args.prefix is not None:
        prefix = args.prefix
    rootpath = prefix+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    createdir(rootpath)

    min_retain_losses = list()
    min_losses = list()

    writejson(f"{rootpath}/settings.json", sys.argv[1:])

    if args.callbacks is not None:
        callbacks = args.callbacks
    else:
        callbacks = list()
    alphalist = [0.8]
    nresults = list()
    for i in range(0,args.n):
        nresults.append(runalldatasets(args, callbacks,
                                       datasetlist, rootpath,
                                       runlist, alphalist=alphalist,
                                       n=i, printcvresults=args.cvsummary,
                                       printcv=args.printcv,
                                       doevaluation=args.doevaluation))
        writejson(f"{rootpath}/data.json", nresults)
    #plotNResults(datasetlist, nresults, rootpath, args.kfold, args.n)
    resdf = pd.DataFrame(nresults)
    resdf.to_csv(f"{rootpath}/results_{args.kfold}kfold_{args.epochs}epochs_{args.onehot}onehot.csv")

    #if "retain_measure" in callbacks:
    #    plotTrainings(datasetlist,dataset_results,rootpath, args.kfold)





def printClassDistr(data, target, index):
    comb = np.zeros((data[index].shape[0], data.shape[1] + 1))
    comb[:, 0:data.shape[1]] = data[index]
    comb[:, data.shape[1]:data.shape[1] + 1] = target[index]
    df = pd.DataFrame(data=comb)
    print(df.iloc[:, data.shape[1]].value_counts())

def CVSearch(model, data, target, args,dsl):
    my_neurons = [15, 30, 60]
    my_epochs = [50, 100, 150]
    my_batch_size = [5, 10]
    my_param_grid = dict(hidden=my_neurons, epochs=my_epochs, batch_size=my_batch_size)

    model2Tune = KerasClassifier(build_fn=makeANNModel(2 * data.shape[1], 1, args.hiddenlayers_gabel, dsl.isregression,
                                           optimizer=None,
                                           onehot=args.onehot, multigpu=args.multigpu), verbose=0)

def writejson(filename,data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)




if __name__ == "__main__":
    main()
