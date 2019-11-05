import argparse
import sys

import keras

from utils.plotting import plotEmbeddingClusters
from utils.runutils import getParser, parseMethod
from dataset.dataset import Dataset
from models.eval import sillouettescore

import matplotlib
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import json as jsonlib
import numpy as np

matplotlib.use('Agg')
# import matplotlib as mpl
# mpl.use('module://backend_interagg')
# https://www.adressa.no/pluss/okonomi/2018/09/08/Prora-eierne-g%C3%A5r-for-frifinnelse-i-ankesaken-17471354.ece



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

def main():
    parser = argparse.ArgumentParser(description='Plot the validation loss per epoch during training.')
    parser.add_argument('--resultdir', metavar='resultdir', type=str,
                        help='resultdir for loading data files including figures',required=True)
    parser.add_argument('--savedir', metavar='savedir', type=str,
                        help='Directory to write the pdf figure to',required=True)
    parser.add_argument('--filename', metavar='filename', type=str,
                        help='Filename for pdf figure',required=True)
    parser.add_argument('-c', '--classes', metavar="classes",
                        help='which classes should be plotted in the clustering visualization.', type=lambda s: [item for item in s.split(',')],
                        dest="classes")
    parser.add_argument('--addgabelresults', metavar='addgabelresults', type=str2bool,
                        help='Filename for pdf figure', default=False)
    parser.add_argument('--removeoptimizer', metavar='removeoptimizer', type=str2bool,
                        help='remove the optimizer name from the method label', default=False)
    parser.add_argument('--doeval', metavar='doeval', type=str2bool,
                        help='Do eval of the model', default=False)
    parser.add_argument('--font_scale', metavar='font_scale', type=float,
                        help='UI scale for the figure.',required=True)
    parser.add_argument('--split', metavar='split', type=int,
                        help='N split for, 5 means 20% will be used for clustering vizualisation.',required=True)
    parser.add_argument('--seed', metavar='seed', type=int,
                        help='random seed',required=True)
    parser.add_argument('--maxdatapoints', metavar='maxdatapoints', type=int,
                        help='maximum number of datapoints to plot',required=True)
    parser.add_argument('--hue_order', metavar='hue_order', type=str,
                        help='Ordering of the different hues in the plot '
                        'so that the methods gets the same hue color in each plot', required=True)
    # parser.add_argument('--modelfile', metavar='modelfile', type=str,
    #                     help='File name for the saved model', required=True)
    parser.add_argument('-modelfiles', '--modelfiles', metavar="modelfiles",
                        help='the model files of the different similiarity measurement methods.', type=lambda s: [item for item in s.split(',')],
                        dest="modelfiles")
    # parser.add_argument('--method', metavar='method', type=str,
    #                     help='method to use', required=True)
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print("not enough arguments")
        parser.print_help()
        sys.exit(1)

    parser = getParser()

    oldargs = None
    with open(f"{args.resultdir}/settings.json") as f:
        oldargv = jsonlib.load(f)

    oldargs = parser.parse_args(args=oldargv)
    #fmethods = oldargs.methods.split(",")
    np.random.seed(args.seed)
    oldmethods = {}
    for method in oldargs.methods:
        runner, oldmethods[method.split(":")[0]] = parseMethod(method)

    dataset = Dataset(oldargs.datasets[0])
    dsl, colmap, \
        stratified_fold_generator = fromDataSetToSKLearn(dataset,
                                                         oldargs.onehot,
                                                         n_splits=args.split)
    train, test = next(stratified_fold_generator)

    features = dsl.getFeatures()
    targets = dsl.getTargets()
    resultfile = args.resultdir
    save_directory = args.savedir
    filename = args.filename
    classes = [int(c) for c in args.classes]
    c = 0
    for method in oldmethods:
        modelmaker = oldmethods[method]["runnerdict"]["makeModel"]
        evalfunc = oldmethods[method]["runnerdict"]["eval"]
        model, embeddingmodel = modelmaker(features, targets,
                                           oldargs.hiddenlayers,
                                           regression=False)
        modelfile = args.modelfiles[c]
        c += 1
        model.load_weights(modelfile)



        randommodel = keras.models.clone_model(embeddingmodel)
        shuffle_weights(randommodel)
        indexes, modeltargets = plotEmbeddingClusters(save_directory, embeddingmodel,
                                                      randommodel,
                                                      filename+method, args.font_scale,
                                                      features, targets, test, classes,
                                                      args.maxdatapoints,
                                                      args.removeoptimizer,
                                                      oldmethods,
                                                      args.hue_order.split(","))
        print(f"did plotting with {len(indexes)} number of indexes")
        if args.doeval:
            print("evaluating the clustering performance using a square of the same indexes")
            #def eval_dual_ann(model, test_data, test_target, train_data,
            #train_target, batch_size, anynominal=False, colmap=None, gabel=False)
            evalscore = sillouettescore(model,
                                        features[indexes],
                                        modeltargets)
            print(f"evalscore: {evalscore}")
        


if __name__ == "__main__":
    main()
