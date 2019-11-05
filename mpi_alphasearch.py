from mpi4py import MPI
from models.model_utils import makeANNModel
from utils.keras_utils import set_keras_growth
from utils.storage_utils import createdir,writejson
from utils.runutils import runalldatasetsMPI,getArgs
from datetime import datetime
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
import random

def main(mpirank, mpisize, mpicomm):
    args = getArgs()
    if args.seed is not None:
        random.seed(args.seed)
    datasetlist = args.datasets
    print(f"doing experiment with {datasetlist} in that order")

    k = args.kfold
    results = {}
    runlist = args.methods

    if "," in args.gpu:
        gpus = args.gpu.split(",")
        mygpu = gpus[mpirank % 2]
        set_keras_growth(int(mygpu))
    else:
        set_keras_growth(args.gpu)

    dataset_results = dict()
    prefix = "runner"
    if args.prefix is not None:
        prefix = args.prefix
    rootpath = prefix+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if mpirank == 0:
        createdir(rootpath)
        writejson(f"{rootpath}/settings.json", vars(args))

    min_retain_losses = list()
    min_losses = list()
    if args.alpharange is not None:
        splits = args.alpharange.split(":")
        alphastart = float(splits[0])
        alphastop = float(splits[1])
        alpharange = np.linspace(alphastart, alphastop, args.alphacount)
    else:
        alpharange = np.linspace(0.000001, 1.00001, args.alphacount)

    datasetsdone = list()
    if args.callbacks is not None:
        callbacks = args.callbacks
    else:
        callbacks = list()
    nresults = list()
    for i in range(0, args.n):
        dataset_results = runalldatasetsMPI(args, callbacks,
                                            datasetlist, mpicomm,
                                            mpirank, rootpath,
                                            runlist, alpharange,
                                            n=i, printcvresults=args.cvsummary,
                                            printcv=args.printcv)
        nresults.append(dataset_results)

        if mpirank == 0:
            # plotNAlphaResults(datasetlist, nresults, rootpath)
            writejson(f"{rootpath}/data.json", nresults)
            resdf = pd.DataFrame(results)
            resdf.to_csv(f"{rootpath}/results_{args.kfold}kfold_{args.epochs}epochs_{args.onehot}onehot.csv")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    mpisize = comm.Get_size()
    mpirank = comm.Get_rank()
    main(mpirank, mpisize, comm)
