import argparse
import sys
from utils.plotting import readJson
from utils.runutils import splitNoEscapes
from utils.plotting import plotNJson
import matplotlib

matplotlib.use('Agg')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='plot results from annSim')
    parser.add_argument('--resultdir', metavar='resultdir', type=str,
                        help='resultdir for loading data files including figures')
    parser.add_argument('--savedir', metavar='savedir', type=str,
                        help='Directory to write the pdf figure to')
    parser.add_argument('--filename', metavar='filename', type=str,
                        help='Filename for pdf figure')
    parser.add_argument('--font_scale', metavar='font_scale', type=float,
                        help='UI scale for the figure.')
    parser.add_argument('--legend_loc', metavar='legend_loc', type=int,
                        help='Where to put the legend in the figure.')
    parser.add_argument('--addgabelresults', metavar='filename', type=str2bool,
                        help='Filename for pdf figure', default=False)
    parser.add_argument('--drawdev', metavar='drawdev', type=str2bool,
                        help='Draw the deviation on the graphs', default=False)
    parser.add_argument('--sortmethodname', metavar='sortmethodname', type=str,
                        help='String (can be substring, e.g. "dees" of "dees:rprop) '
                             'of which methods values to sort the graph on', required=True)
    parser.add_argument('--removeoptimizer', metavar='removeoptimizer', type=str2bool,
                        help='remove the optimizer name from the method label', default=False)
    parser.add_argument('--hue_order', metavar='hue_order', type=str,
                        help='Ordering of the different hues in the plot '
                             'so that the methods gets the same hue color in each plot', required=True)
    parser.add_argument('--dataset_order', metavar='dataset_order',
                        help='Ordering of the different datasets (x-axis) in the plot ',
                        type=lambda s: [item for item in s.split(',')],
                        dest="dataset_order")

    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print ("not enough arguments")
        parser.print_help()
        sys.exit(1)
    sortmethodname = args.sortmethodname
    if args.removeoptimizer:
        if ":" in sortmethodname:
            sortmethodname = args.sortmethodname.split(":")[0]
    plotNJson(args.resultdir, 5, args.savedir,
              args.filename, sortmethodname,
              args.addgabelresults, args.drawdev, args.font_scale,
              args.removeoptimizer, splitNoEscapes(args.hue_order, ","),
              args.dataset_order, args.legend_loc)


if __name__ == "__main__":
    main()
