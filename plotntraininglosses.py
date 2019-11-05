import argparse
import sys
from utils.plotting import readJson, plotNTrainings
from utils.runutils import getParser,parseMethod
from utils.runutils import splitNoEscapes
from dataset.dataset import Dataset
import matplotlib


matplotlib.use('Agg')
# import matplotlib as mpl
# mpl.use('module://backend_interagg')
# https://www.adressa.no/pluss/okonomi/2018/09/08/Prora-eierne-g%C3%A5r-for-frifinnelse-i-ankesaken-17471354.ece
import json as jsonlib


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Plot the validation loss per epoch during training.')
    parser.add_argument('--resultdir', metavar='resultdir', type=str,
                        help='resultdir for loading data files including figures',required=True)
    parser.add_argument('--savedir', metavar='savedir', type=str,
                        help='Directory to write the pdf figure to',required=True)
    parser.add_argument('--filename', metavar='filename', type=str,
                        help='Filename for pdf figure',required=True)
    parser.add_argument('--addgabelresults', metavar='addgabelresults', type=str2bool,
                        help='Filename for pdf figure', default=False)
    parser.add_argument('--legend_loc', metavar='legend_loc', type=int,
                        help='Where to put the legend in the figure.')
    parser.add_argument('--removeoptimizer', metavar='removeoptimizer', type=str2bool,
                        help='remove the optimizer name from the method label', default=False)
    parser.add_argument('--font_scale', metavar='font_scale', type=float,
                        help='UI scale for the figure.',required=True)
    parser.add_argument('--showevals', metavar='showevals', type=str2bool,
                        help='Use evaluation rather than epochs for x axis.')
    parser.add_argument('--pdf', metavar='pdf', type=str2bool,
                        help='PDF instead of EPS.')
    parser.add_argument('--hue_order', metavar='hue_order', type=str,
                        help='Ordering of the different hues in the plot '
                             'so that the methods gets the same hue color in each plot', required=True)
    parser.add_argument('--filtermethods', metavar='filtermethods', type=str,
                        help='Ordering of the different hues in the plotNTrainings '
                        'so that the methods gets the same hue color in each plot', default=None)
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print("not enough arguments")
        parser.print_help()
        sys.exit(1)

    parser2 = getParser()

    oldargs = None
    with open(f"{args.resultdir}/settings.json") as f:
        oldargv = jsonlib.load(f)

    oldargs = parser2.parse_args(args=oldargv)
    #fmethods = oldargs.methods.split(",")
    
    oldmethods = {}
    for method in oldargs.methods:
        runner, oldmethods[method.split(":")[0]] = parseMethod(method)

    datasetsize = Dataset(oldargs.datasets[0]).getNumberOfRows()*0.8
    resultfile = args.resultdir
    json = readJson(resultfile)
    save_directory = args.savedir
    filename = args.filename

    if args.filtermethods is None:
        filtermethods = ""
    else:
        filtermethods = args.filtermethods

    plotNTrainings(json, save_directory, filename, args.font_scale,
                   args.removeoptimizer, oldargs.showevals or args.showevals,
                   oldmethods, datasetsize, 
                   splitNoEscapes(args.hue_order, ","),
                   splitNoEscapes(filtermethods, ","), 
                   pdf=args.pdf, legend_loc=args.legend_loc)


if __name__ == "__main__":
    main()
