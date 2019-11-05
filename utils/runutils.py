import time
import numpy as np
from keras.optimizers import Adam, RMSprop

from dataset.dataset import Dataset
from models.type2 import makeGabelArch
from models.type4 import makeDualArch, makeEndToEndDualArch, makeEndToEndDualArchShared, dees_resnet
from models.type3 import makeNormalArch, chopra, make_chopra_model
from models.esnn import esnn, make_eSNN_model
from models.eval import eval_normal_ann_l2, eval_dual_ann, \
    eval_gabel_ann, eval_chopra_ann
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.rprop import RProp
from models.type1 import sim_def_lin, sim_def_nonlin
from dataset.makeTrainingData import makeDualSharedArchData,\
    makeGabelTrainingData, makeNData, makeSmartNData, makeSemiBig
# from utils.plotting import plotResults3
from utils.storage_utils import writejson, createdir
from utils.KerasCallbacks import callbackdict
import argparse
import sys
import hashlib

def splitNoEscapes(string, char):
    if string is None or len(string) < 2:
        return [string]
    sections = string.split(char)
    sections = [i + (char if i[-1] == "\\" else "") for i in sections]
    result = ["" for i in sections]
    j = 0
    for s in sections:
        result[j] += s
        j += (1 if s[-1] != char else 0)
    ret = [i.replace("\\", "") for i in result if i != ""]
    return ret

""" This dict contains methods with the method name as key and another dict as
value, this method dictionary should specify the method that constructs and
trains the model, and the method that evaluates the trained model """
rundict = {
    "gabel": {"modeltrainer": makeGabelArch, "eval": eval_gabel_ann, "figname": "$gabel$"},
    "eSNN": {"modeltrainer": esnn, "makeModel": make_eSNN_model, "eval": eval_dual_ann, "figname": "$eSNN$"},
    "t3i1": {"modeltrainer": makeNormalArch, "eval": eval_normal_ann_l2, "figname": "$t_{3,1}$"},
    "t4i4": {"modeltrainer": makeDualArch, "eval": eval_dual_ann, "figname": "$t_{4,4}$"},
    "t4i1": {"modeltrainer": makeEndToEndDualArch, "eval": eval_dual_ann, "figname": "$t_{4,1}$"},
    "t4i2": {"modeltrainer": makeEndToEndDualArchShared, "eval": eval_dual_ann, "figname": "$t_{4,2}$"},
    "t4i6": {"modeltrainer": dees_resnet, "eval": eval_dual_ann, "figname": "$t_{4,6}$"},
    "t1i1": {"modeltrainer": None, "eval": sim_def_lin, "figname": "$t_{1,1}$"},
    "t2i1": {"modeltrainer": None, "eval": sim_def_nonlin, "figname": "$t_{2,1}$"},
    "chopra": {"modeltrainer":
                 chopra,
                 "makeModel": make_chopra_model,
                 "eval": eval_chopra_ann, "figname": "$chopra$"}
}

non_trainable = ["t1i1", "t2i1"]
optimizer_dict = {
    "rprop": {"constructor": RProp, "batch_size": "full"},
    "adam": {"constructor": Adam, "batch_size":None},
    "rmsprop": {"constructor": RMSprop, "batch_size":None}
}
maketrainingdata_dict = {
    "gabel": {"func": makeGabelTrainingData, "factorfunc": lambda n: n**2},
    "split": {"func": makeDualSharedArchData, "factorfunc": lambda n: n**2},
    "ndata": {"func": makeNData, "factorfunc": lambda n: n},
    "smartndata": {"func": makeSmartNData, "factorfunc": lambda n: n},
    "halfbig": {"func": makeSemiBig, "factorfunc": lambda n: ((n**2)/2)-n}
}


def parseMethod(methodstring):
    retdict = {}
    runnersplit = None
    methodname = ""
    if methodstring not in non_trainable:
        runner_split = splitNoEscapes(methodstring,":")
        methodname = runner_split[0]
    else:
        runnerdict = rundict[methodstring]
        methodname = methodstring
    if runner_split:
        runnerdict = rundict[runner_split[0]]
        if len(runner_split) > 1:
            optimizer = optimizer_dict[runner_split[1]]
        if len(runner_split) > 2:
            retdict["epochs"] = int(runner_split[2])
        if len(runner_split) > 3:
            retdict["maketrainingdata"] = maketrainingdata_dict[runner_split[3]]
        if len(runner_split) > 4:
            retdict["alpha"] = float(runner_split[4])

    retdict["runnerdict"] = runnerdict

    return methodname, retdict

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getParser():
    parser = argparse.ArgumentParser(description='train NN classification'
                                     + ' model on datasets against CBR!')
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        help='How many epochs')
    parser.add_argument('--evals', metavar='evals', type=int,
                        help='How many evaluations should be used'\
                        'in training (epochs*trainingexamples)')
    parser.add_argument('--showevals', metavar='True/False',
                        type=str2bool,
                        help='To show number of evals instead of epochs in results.',
                        default=False)
    parser.add_argument('--kfold', metavar='kfold', type=int,
                        help='How many folds in cross validation',
                        default=None)
    parser.add_argument('--n', metavar='n', type=int,
                        help='How many folds in cross validation',
                        default=None)
    parser.add_argument('--gpu', metavar='gpu', type=str,
                        help='which gpu should be used',default="")
    parser.add_argument('--prefix', metavar='prefix', type=str,
                        help='Prefix for saving data files including figures')
    parser.add_argument('--test', metavar='test', type=str,
                        help='')
    parser.add_argument('--onehot', metavar='True/False', type=str2bool,
                        help='Use onehot encoding (one col for each value)',
                        default=True)
    parser.add_argument('--multigpu', metavar='True/False',
                        type=str2bool,
                        help='Enable multigpu support.',
                        default=True)
    parser.add_argument('--cvsummary', metavar='True/False', type=str2bool,
                        help='Enable detailed cv fold output.', default=True)
    parser.add_argument('--printcv', metavar='True/False', type=str2bool,
                        help='Print CV indexes.', default=True)
    parser.add_argument('--doevaluation', metavar='True/False', type=str2bool,
                        help='Do evaluation after training.', default=True)
    parser.add_argument('--batchsize', metavar='batchsize', type=int,
                        help='Size of batch to use in training', default=32)
    parser.add_argument('--alphacount', metavar='alphacount', type=int,
                        help='number of steps when testing alphas', default=32)
    parser.add_argument('-alpharange', '--alpharange',metavar="alpharange" ,
                        help='<rangestart:rangeend>',
                        type=str)
    parser.add_argument('--seed', metavar='seed', type=int,
                        help='Size of batch to use in training')
    parser.add_argument('-ds', '--datasets',metavar="datasets",
                        help='comma delimited list of datasets',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-cb', '--callbacks',metavar="callbacks" ,
                        help=f'comma delimited list of callbacks: {str(callbackdict.keys())}',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-ms', '--methods',metavar="methods" ,
                        help='comma delimited list of methods with the possible extension as such: <method:optimizer:epochs:trainingdata:alpha>',
                        type=lambda s: [item for item in splitNoEscapes(s,',')])
    parser.add_argument('-hi', '--hiddenlayers',
                        metavar="hiddenlayers",
                        help='comma delimited list of hidden layer sizes for '
                        +'gabel network, e.g 3,3 for two hidden layers of 3 neurons',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-c', '--c_layers',
                        metavar="c_layers",
                        help='comma delimited list of hidden layer sizes for '
                        +'c(.,.)',
                        type=str, nargs='+', action='append')
    parser.add_argument('-g', '--g_layers',
                        metavar="g_layers",
                        help='comma delimited list of hidden layer sizes for '
                        +'g(.)',
                        type=str, nargs='+', action='append')

    return parser


def getArgs():
    parser = getParser()
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print ("not enough arguments")
        parser.print_help()
        sys.exit(1)

    if args.datasets is None  or args.methods is None:
        print("you must supply a list of datasets and a list of methods to run this program")
        parser.print_help()
        sys.exit(1)

    if args.hiddenlayers is not None:
        newlist = []
        if args.c_layers is not None:
            newlist = [args.hiddenlayers, args.c_layers[0]]
        else:
            newlist = [args.hiddenlayers, []]
        args.hiddenlayers = newlist
    elif args.hiddenlayers is None and args.g_layers is not None:
        args.hiddenlayers = [args.g_layers[0], args.c_layers[0]]
    else:
        print("you must supply a list of datasets and a list of methods to run this program")
        parser.print_help()
        sys.exit(1)
    print(f"args.hiddenlayers {args.hiddenlayers} ")
    return args


def runalldatasets(args, callbacks, datasetlist, rootpath,
                   runlist, alphalist, n, printcvresults=False,
                   printcv=False, doevaluation=True, evalcount=False):
    dataset_results = dict()
    datasetsdone = list()
    totalfolds = args.kfold
    for dataset in datasetlist:
        d = Dataset(dataset)
        datasetsdone.append(dataset)
        # d = ds.getDataset(dataset)
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, args.onehot, n_splits=args.kfold)
        data = dsl.getFeatures()
        target = dsl.getTargets()

        anynominal = False
        datasetInfo = dsl.dataset.datasetInfo
        if not args.onehot:
            for key, value in colmap.items():
                isclass = False
                for col in datasetInfo["cols"]:
                    if col["name"] == key and col["class"] == True:
                        isclass = True
                    if value["type"] is "nominal" and not isclass:
                        anynominal = True

        # kfold = 0
        min_retain_losses = list()
        min_losses = list()
        dataset_result = dict()
        fold_number = 0
        # for each split in the kfold validation
        for train, test in stratified_fold_generator:
            fold_number += 1
            fold_results = dict()
            if printcv:
                trainstr = "-".join(str(x) for x in train)
                trainhash = hashlib.md5(trainstr.encode()).digest()

                teststr = "-".join(str(x) for x in test)
                testhash = hashlib.md5(teststr.encode()).digest()

                print(f"summary of this cv-fold, first train: {train[0]} trainhash: {trainhash}"
                      f"first test: {test[0]} testhash: {testhash}")

            # dataset_results[str(fold_number)] = fold_results
            # run all the methods in the split, so we can compare them internally
            # later if they are within each others standard deviation
            fold_results, ranlist = rundataset(anynominal, args, callbacks,
                                               colmap, data, dataset,
                                               dsl, fold_results, min_losses,
                                               min_retain_losses, rootpath,
                                               runlist, target, test,
                                               train, alphalist=alphalist,
                                               printcvresults=printcvresults,
                                               n=n, doevaluation=doevaluation,
                                               fold_number=fold_number)
            dataset_result[str(fold_number)] = fold_results

        dataset_results[dataset] = dataset_result
        # if not args.mlp:
        writejson(f"{rootpath}/n{n}-data.json", dataset_results)
        #writejson(f"{rootpath}/data.json", dataset_results)
        #plotResults3(datasetsdone, dataset_results, rootpath, args.kfold)
        printSummary(dataset, dataset_result, runlist, n, args)
        # modelsize = get_model_memory_usage(args.gabel_batchsize,gabel_model)
        # model size is in GB, so setting this as gpu fraction is 12 x what we need..
        # set_keras_parms(threads=0,gpu_fraction=modelsize)
    return dataset_results

def runalldatasetsMPI(args, callbacks,
                      datasetlist, mpicomm,
                      mpirank, rootpath,
                      runlist, alphalist, n,
                      printcvresults=False,
                      printcv=False,
                      doevaluation=True,
                      evalcount=False):
    datasetsdone = list()
    dataset_results = dict()
    for dataset in datasetlist:
        d = Dataset(dataset)
        datasetsdone.append(dataset)
        # d = ds.getDataset(dataset)
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, args.onehot, n_splits=args.kfold)
        data = dsl.getFeatures()
        target = dsl.getTargets()

        anynominal = False
        datasetInfo = dsl.dataset.datasetInfo
        if not args.onehot:
            for key, value in colmap.items():
                isclass = False
                for col in datasetInfo["cols"]:
                    if col["name"] == key and col["class"] == True:
                        isclass = True
                    if value["type"] is "nominal" and not isclass:
                        anynominal = True

        # kfold = 0
        min_retain_losses = list()
        min_losses = list()
        dataset_result = dict()
        fold_number = 0
        # for each split in the kfold validation
        foldlist = list()
        for train, test in stratified_fold_generator:
            foldlist.append((train, test))
        mpicomm.barrier()

        train, test = foldlist[mpirank]
        if printcv:
            trainstr = "-".join(str(x) for x in train)
            trainhash = hashlib.md5(trainstr.encode()).digest()

            teststr = "-".join(str(x) for x in test)
            testhash = hashlib.md5(teststr.encode()).digest()

            print(f"summary of this cv-fold, first train: {train[0]} trainhash: {trainhash}"
                  f"first test: {test[0]} testhash: {testhash}")
        fold_number = mpirank
        fold_results = dict()
        mpi_rank_rootpath = f"{rootpath}/{mpirank}"
        createdir(mpi_rank_rootpath)
        # dataset_results[str(fold_number)] = fold_results,
        # run all the methods in the split, so we can compare them internally
        # later if they are within each others standard deviation
        fold_results, ranlist = rundataset(anynominal, args, callbacks, colmap,
                                           data, dataset, dsl, fold_results,
                                           min_losses, min_retain_losses,
                                           mpi_rank_rootpath, runlist, target,
                                           test, train, alphalist=alphalist,
                                           printcvresults=printcvresults, n=n,
                                           doevaluation=doevaluation,
                                           fold_number=fold_number)
        mpicomm.barrier()
        if mpirank == 0:
            dataset_result[str(fold_number)] = fold_results
            for i in range(1, args.kfold):
                recv_fold_results = mpicomm.recv(source=i)
                dataset_result[i] = recv_fold_results
        elif mpirank != 0:
            mpicomm.send(fold_results, dest=0)

        if mpirank == 0:
            dataset_results[dataset] = dataset_result
            # if not args.mlp:
            writejson(f"{rootpath}/n{n}-data.json", dataset_results)
            #if len(alphalist) == 1:
            #    plotResults3(datasetsdone, dataset_results, rootpath, args.kfold)
            printSummary(dataset, dataset_result, ranlist, n, args)
        # modelsize = get_model_memory_usage(args.gabel_batchsize,gabel_model)
        # model size is in GB, so setting this as gpu fraction is 12 x what we need..
        # set_keras_parms(threads=0,gpu_fraction=modelsize)
    return dataset_results

def rundataset(anynominal, args, callbacks, colmap, data,
               dataset, dsl, fold_results, min_losses,
               min_retain_losses, rootpath, runlist,
               target, test, train, alphalist, printcvresults=False,
               n=0, doevaluation=True, fold_number=-1, evalcount=False):
    ranlist = list()
    alphaiterations = False
    if len(runlist) == 1 and len(alphalist) > 1:
        alphaiterations = True
    for runner in runlist:
        for alpha in alphalist:
            ranmethod = iteration(anynominal, args, callbacks,
                                  colmap, data,
                                  dataset, dsl, fold_results,
                                  min_losses, min_retain_losses,
                                  rootpath, runner, target,
                                  test, train, alpha,
                                  alphaiterations, printcvresults,
                                  n, alphatest=False,
                                  doevaluation=doevaluation,
                                  fold_number=fold_number)
            ranlist.append(ranmethod)
    return fold_results, ranlist


def iteration(anynominal, args, callbacks, colmap, data,
              dataset, dsl, fold_results, min_losses,
              min_retain_losses, rootpath, runner,
              target, test, train, alpha,
              alphaiterations=False, printcvresults=False,
              n=0, alphatest=False, doevaluation=True,
              fold_number=-1, evalcount=False):
    runner_split = None
    epochs = args.epochs
    makeTrainingDataKey = "split"
    gabel = False
    if "gabel" in runner:
        gabel = True
        makeTrainingDataKey = "gabel"
    #makeTrainingData = maketrainingdata_dict[makeTrainingDataKey]["func"]
    if runner not in non_trainable:
        runner_split = runner.split(":")
    else:
        runnerdict = rundict[runner]
    if runner_split:
        runnerdict = rundict[runner_split[0]]
        if len(runner_split) > 1:
            optimizer = optimizer_dict[runner_split[1]]
        else:
            optimizer = RProp
        if len(runner_split) > 2:
            epochs = int(runner_split[2])
        if len(runner_split) > 3:
            makeTrainingDataKey = runner_split[3]
        if len(runner_split) > 4:
            alpha = float(runner_split[4])
    makeTrainingData = maketrainingdata_dict[makeTrainingDataKey]
    makeTrainingDataFunc = makeTrainingData["func"]
    #if evalcount:
    #    epoch = 
    trainer = runnerdict["modeltrainer"]
    eval_func = runnerdict["eval"]

    if not alphaiterations:
        methodstring = runner
    else:
        methodstring = str(alpha)
    fold_results[methodstring] = dict()
    model = None
    hist = None
    time_start = time.time()
    if trainer is not None:
        model, hist, \
        ret_callbacks, embedding_model = trainer(o_X=data[test], o_Y=target[test],
                                                 X=data[train], Y=target[train],
                                                 regression=dsl.isregression,
                                                 shuffle=True,
                                                 batch_size=args.batchsize,
                                                 epochs=epochs,
                                                 optimizer=optimizer,
                                                 onehot=args.onehot,
                                                 multigpu=args.multigpu,
                                                 callbacks=callbacks,
                                                 datasetname=dataset,
                                                 networklayers=args.hiddenlayers,
                                                 rootdir=rootpath,
                                                 alpha=alpha,
                                                 makeTrainingData=makeTrainingDataFunc)

        min_loss = hist.history["loss"][len(hist.history["loss"]) - 1]
        fold_results[methodstring]["training_losses"] = hist.history["loss"]
        fold_results[methodstring]["training_loss"] = min_loss
        fold_results[methodstring]["epochs"] = len(hist.history["loss"])
        if evalcount:
            fold_results[methodstring]["evals"] = maketrainingdata_dict["factorfunc"](len(hist.history["loss"]))
        min_losses.append(min_loss)
        if "retain_measure" in callbacks:
            training_losses = ret_callbacks["retain_measure"].losses
            training_retain_losses = ret_callbacks["retain_measure"].retain_loss
            fold_results[methodstring]["training_loss_hist"] = training_losses
            fold_results[methodstring]["training_retain_loss_hist"] = training_retain_losses
    if methodstring not in non_trainable or "gabelelitism" not in callbacks:
        # min_recall_loss = np.min(callback.recall_loss)
        min_retain_loss = 1
        if doevaluation:
            res = eval_func(model, data[test], target[test], data[train],
                            target[train], batch_size=args.batchsize,
                            anynominal=anynominal, colmap=colmap)

            acc = np.sum(res) / len(res)
            min_retain_loss = 1.0 - acc
        min_retain_losses.append(min_retain_loss)
    elif methodstring not in non_trainable:
        min_retain_loss = ret_callbacks["gabelelitism"].best
    fold_results[methodstring]["ret_loss"] = min_retain_loss
    timespent = time.time() - time_start
    fold_results[methodstring]["timespent"] = timespent
    if printcvresults:
        printCVSummary(dataset, fold_results, methodstring,
                       n, args, fold_number)
    # print(f"dataset {dataset} method {methodstring}, "
    #      f"min_retrieve_loss: {min_retain_loss} time spent: {timespent}"
    return methodstring


def printSummary(dataset, dataset_result, runlist, n, args):
    for method in runlist:
        ret_results = np.zeros((len(dataset_result.keys()), 1))
        if method not in non_trainable:
            training_results = np.zeros((len(dataset_result.keys()), 1))
            epochs = np.zeros((len(dataset_result.keys()), 1))
            timespentarr = np.zeros((len(dataset_result.keys()), 1))
        i = 0
        for key in dataset_result:
            ret_results[i] = dataset_result[key][method]["ret_loss"]
            if  method not in non_trainable:
                training_results[i] = dataset_result[key][method]["training_loss"]
                epochs[i] = dataset_result[key][method]["epochs"]
                timespentarr[i] = dataset_result[key][method]["timespent"]
            i += 1

        total_avg_retain_loss = np.mean(ret_results)
        total_std_retain_loss = np.std(ret_results)

        if method not in non_trainable:
            total_avg_epochs = np.mean(epochs)
            total_std_epochs = np.std(epochs)
            total_avg_timespent = np.mean(timespentarr)
            total_std_timespent = np.std(timespentarr)
            total_avg_training_loss = np.mean(training_results)
            total_std_training_loss = np.std(training_results)
            print(f"n:{n}/{args.n-1} "
                  f"dataset {dataset} method {method} "
                  f"avg_retrieve_loss: {total_avg_retain_loss} "
                  f"std_retrieve_loss: {total_std_retain_loss} "
                  f"avg_training_loss: {total_avg_training_loss} "
                  f"std_training_loss: {total_std_training_loss} "
                  f"epochs: {total_avg_epochs} (+/- {total_std_epochs}) "
                  f"retloss/epoch: {total_avg_retain_loss/total_avg_epochs} "
                  f"avg_timespent: {total_avg_timespent} (+/- {total_std_timespent}) "
                  )
        else:
            print(f"dataset {dataset} method {method} "
                  f"avg_retrieve_loss: {total_avg_retain_loss} "
                  f"std_retrieve_loss: {total_std_retain_loss}"
                  )

def printCVSummary(dataset, CV_result, method, n, args, fold_number):
    retain_loss = CV_result[method]["ret_loss"]
    if method not in non_trainable:
        training_loss = CV_result[method]["training_loss"]
        epochs = CV_result[method]["epochs"]
        timespent = CV_result[method]["timespent"]

    if method not in non_trainable:
        print(f"n:{n}/{args.n-1} "
              f"cv: {fold_number}/{args.kfold} "
              f"dataset {dataset} method {method} "
              f"retrieve_loss: {retain_loss} "
              f"training_loss: {training_loss} "
              f"epochs: {epochs} "
              f"retloss/epoch: {retain_loss/epochs} "
              f"avg_timespent: {timespent} "
        )
    else:
        print(f"dataset {dataset} method {method} "
              f"avg_retrieve_loss: {retain_loss} "
        )
