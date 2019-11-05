import argparse
import sys
import json
from utils.plotting import  plotJson

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
    parser.add_argument('--addgabelresults', metavar='filename', type=str2bool,
                        help='Filename for pdf figure',default=False)
    parser.add_argument('--drawdev', metavar='drawdev', type=str2bool,
                        help='Draw the deviation on the graphs',default=False)
    args = parser.parse_args()


    

    if not len(sys.argv) > 1:
        print ("not enough arguments")
        parser.print_help()
        sys.exit(1)

    plotJson(args.resultdir,5,args.savedir,args.filename,args.addgabelresults,args.drawdev)

if __name__ == "__main__":
    main()
