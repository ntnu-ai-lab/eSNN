from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.model_utils import makeANNModel, \
    makeDualEndToEndArchSingleHead
from models.type2 import makeGabelArch
from models.type4 import makeDualArch, makeEndToEndDualArch, makeEndToEndDualArchShared
from models.type3 import makeNormalArch
from utils.KerasCallbacks import callbackdict
from dataset.dataset import Dataset
from utils.keras_utils import set_keras_growth
from utils.storage_utils import createdir
from utils.plotting import plotAlphaResults
from keras.optimizers import Adam,RMSprop
from models.rprop import RProp
from datetime import datetime
from models.eval import eval_dual_ann, eval_normal_ann,\
    eval_gabel_ann
from models.type1 import sim_def_lin, sim_def_nonlin
from keras.wrappers.scikit_learn import KerasClassifier
import argparse
import sys
import numpy as np
import pandas as pd
import json

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

defaulthost="localhost:8080"
myCBRpath = "/Users/epic/research/papers/iccbr2018/code/mycbr-rest/target/mycbr-rest-1.0-SNAPSHOT.jar"

rundict = {
    "gabel": {"makeModel": makeGabelArch, "eval": eval_dual_ann},
    "gabel": {"makeModel": makeGabelArch, "eval": eval_gabel_ann},
    "normal": {"makeModel": makeNormalArch, "eval": eval_normal_ann},
    "dualarch": {"makeModel": makeDualArch, "eval": eval_dual_ann},
    "dee": {"makeModel": makeEndToEndDualArch, "eval": eval_dual_ann},
    "dees": {"makeModel": makeEndToEndDualArchShared, "eval": eval_dual_ann},
    "deesa": {"makeModel": makeDualEndToEndArchSingleHead, "eval": eval_dual_ann},
    "sim_def_lin": {"makeModel":None, "eval": sim_def_lin},
    "sim_def_nonlin": {"makeModel":None, "eval": sim_def_nonlin}
}
optimizer_dict = {
    "rprop": {"constructor": RProp, "batch_size": "full"},
    "adam": {"constructor": Adam, "batch_size":None},
    "rmsprop": {"constructor": RMSprop, "batch_size":None}
}
def main():
    parser = argparse.ArgumentParser(description='train NN classification'
                                     + ' model on datasets against CBR!')
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        help='How many epochs')
    parser.add_argument('--kfold', metavar='kfold', type=int,
                        help='How many folds in cross validation',default=None)
    parser.add_argument('--gpu', metavar='gpu', type=int,
                        help='which gpu should be used',default=1)
    parser.add_argument('--prefix', metavar='prefi', type=str,
                        help='Prefix for saving data files including figures')
    parser.add_argument('--onehot', metavar='True/False', type=str2bool,
                        help='Use onehot encoding (one col for each value)', default=True)
    parser.add_argument('--multigpu', metavar='True/False', type=str2bool,
                        help='Enable multigpu support.', default=True)
    parser.add_argument('--batchsize', metavar='batchsize', type=int,
                        help='Size of batch to use in training',default=32)
    parser.add_argument('-ds', '--datasets',metavar="datasets" ,
                        help='comma delimited list of datasets',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-cb', '--callbacks',metavar="callbacks" ,
                        help=f'comma delimited list of callbacks: {str(callbackdict.keys())}',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-ms', '--methods',metavar="methods" ,
                        help='comma delimited list of methods',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-hi', '--hiddenlayers',
                        metavar="hiddenlayers",
                        help='comma delimited list of hidden layer sizes for '
                        +'gabel network, e.g 3,3 for two hidden layers of 3 neurons',
                        type=lambda s: [item for item in s.split(',')])

    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print ("not enough arguments")
        parser.print_help()
        sys.exit(1)

    if args.datasets is None  or args.methods is None:
        print("you must supply a list of datasets and a list of methods to run this program")
        parser.print_help()
        sys.exit(1)

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

    dataset_results = dict()
    prefix = "runner"
    if args.prefix is not None:
        prefix = args.prefix
    rootpath = prefix+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    createdir(rootpath)

    min_retain_losses = list()
    min_losses = list()

    writejson(f"{rootpath}/settings.json", sys.argv[1:])
    datasetsdone = list()
    if args.callbacks is not None:
        callbacks = args.callbacks
    else:
        callbacks = list()
    alpharange = np.linspace(0.10,1.0,args.alphacount)
    methodlist = list()
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

        #kfold = 0
        min_retain_losses = list()
        min_losses = list()
        dataset_result = dict()
        fold_number = 0
        # for each split in the kfold validation
        for train, test in stratified_fold_generator:
            fold_number += 1
            fold_results = dict()
            #dataset_results[str(fold_number)] = fold_results
            # run all the methods in the split, so we can compare them internally
            # later if they are within each others standard deviation
            runner = runlist[0]
            runner_split = runner.split(":")
            runnerdict = rundict[runner_split[0]]
            for alpha in alpharange:
                if len(runner_split) > 1:
                    optimizer = optimizer_dict[runner_split[1]]
                maker = runnerdict["makeModel"]
                eval_func = runnerdict["eval"]
                methodstring = str(round(alpha,2))
                if methodstring not in methodlist:
                    methodlist.append(methodstring)
                fold_results[methodstring] = dict()
                model = None
                hist = None
                if maker is not None:
                    model,hist,ret_callbacks = maker(data[test], target[test],
                                                     data[train], target[train],
                                                     regression=dsl.isregression,
                                                     shuffle=True, batch_size=args.batchsize,
                                                     epochs=args.epochs, optimizer=optimizer,
                                                     onehot=args.onehot, multigpu=args.multigpu,
                                                     callbacks=callbacks, datasetname=dataset,
                                                     networklayers=args.hiddenlayers,
                                                     rootdir=rootpath, alpha=alpha)
                    min_loss = hist.history["loss"][len(hist.history["loss"]) - 1]
                    fold_results[methodstring]["training_loss"] = min_loss
                    fold_results[methodstring]["epochs"] = len(hist.history["loss"])
                    min_losses.append(min_loss)
                    if "retain_measure" in callbacks:
                        training_losses = ret_callbacks["retain_measure"].losses
                        training_retain_losses = ret_callbacks["retain_measure"].recall_loss
                        fold_results[methodstring]["training_loss_hist"] = training_losses
                        fold_results[methodstring]["training_retain_loss_hist"] = training_retain_losses
                if "sim_def" in methodstring or "gabelelitism" not in callbacks:
                    #min_recall_loss = np.min(callback.recall_loss)
                    res = eval_func(model, data[test], target[test],
                    data[train], target[train],
                    batch_size=args.batchsize,
                    anynominal=anynominal,
                    colmap=colmap)

                    acc = np.sum(res) / len(res)
                    min_retain_loss = 1.0-acc
                    min_retain_losses.append(min_retain_loss)
                elif "sim_def" not in methodstring:
                    min_retain_loss = ret_callbacks["gabelelitism"].best
                fold_results[methodstring]["ret_loss"] = min_retain_loss
                print(f"dataset {dataset} method {methodstring}, "
                       f"min_retrieve_loss: {min_retain_loss} ")
            dataset_result[str(fold_number)] = fold_results

        dataset_results[dataset] = dataset_result
        #if not args.mlp:
        writejson(f"{rootpath}/data.json",dataset_results)
        plotAlphaResults(datasetsdone, dataset_results, rootpath, args.kfold)
        for method in methodlist:
            ret_results = np.zeros((len(dataset_result.keys()),1))
            if "sim_def" not in method:
                training_results = np.zeros((len(dataset_result.keys()),1))
                epochs = np.zeros((len(dataset_result.keys()), 1))
            i=0
            for key in dataset_result:
                ret_results[i] = dataset_result[key][method]["ret_loss"]
                if "sim_def" not in method:
                    training_results[i] = dataset_result[key][method]["training_loss"]
                    epochs[i] = dataset_result[key][method]["epochs"]
                i += 1

            total_avg_retain_loss = np.mean(ret_results)
            total_std_retain_loss = np.std(ret_results)

            if "sim_def" not in method:
                total_avg_epochs = np.mean(epochs)
                total_std_epochs = np.std(epochs)
                total_avg_training_loss = np.mean(training_results)
                total_std_training_loss = np.std(training_results)
                print(f"dataset {dataset} method {method} "
                      f"avg_retrieve_loss: {total_avg_retain_loss} "
                      f"std_retrieve_loss: {total_std_retain_loss} "
                      f"avg_training_loss: {total_avg_training_loss} "
                      f"std_training_loss: {total_std_training_loss} "
                      f"epochs: {total_avg_epochs} (+/- {total_std_epochs}) "
                      f"retloss/epoch: {total_avg_retain_loss/total_avg_epochs} "
                      )
            else:
                print(f"dataset {dataset} method {method} "
                      f"avg_retrieve_loss: {total_avg_retain_loss} "
                      f"std_retrieve_loss: {total_std_retain_loss}"
                      )
        # modelsize = get_model_memory_usage(args.gabel_batchsize,gabel_model)
        # model size is in GB, so setting this as gpu fraction is 12 x what we need..
        # set_keras_parms(threads=0,gpu_fraction=modelsize)

    plotAlphaResults(datasetlist, dataset_results, rootpath, args.kfold)
    resdf = pd.DataFrame(results)
    resdf.to_csv(f"{rootpath}/results_{args.kfold}kfold_{args.epochs}epochs_{args.onehot}onehot.csv")

def printClassDistr(data,target,index):
    comb = np.zeros((data[index].shape[0], data.shape[1] + 1))
    comb[:, 0:data.shape[1]] = data[index]
    comb[:, data.shape[1]:data.shape[1] + 1] = target[index]
    df = pd.DataFrame(data=comb)
    print(df.iloc[:, data.shape[1]].value_counts())

def CVSearch(model,data,target,args,dsl):
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
