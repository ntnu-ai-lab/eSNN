import argparse
import sys
import json
from utils.plotting import  plotJson,plotAlphaResults,readJson,jsonToDF
import matplotlib
matplotlib.use('Agg')
#import matplotlib as mpl
#mpl.use('module://backend_interagg')
#https://www.adressa.no/pluss/okonomi/2018/09/08/Prora-eierne-g%C3%A5r-for-frifinnelse-i-ankesaken-17471354.ece
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


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
    args = parser.parse_args()


    

    if not len(sys.argv) > 1:
        print ("not enough arguments")
        parser.print_help()
        sys.exit(1)
    resultfile = args.resultdir
    json = readJson(resultfile)
    save_directory = args.savedir
    filename = args.filename
    df = jsonToDF(json)
    grp = df.groupby(["method"],as_index=False)
    #df["dataset_mean"] = grp.value.transform("mean")
    df["method_mean"] = grp.value.transform("mean")
    #print(grp.head(10))
    #sorted_df = grp.sort_values(by=["method_mean","method"])
    fig, ax = plt.subplots()
    sns.set(font_scale=0.4)
    sns_plot = sns.lineplot(x="method",y="method_mean",data=df)
    # fig, ax = plt.subplots()
    # sns.set(font_scale=0.4)
    # sns_plot = sns.lineplot(x="method",y="value",data=df,sort=False)
    ax.set(xlabel='alpha', ylabel='retrieval loss')
    fig = sns_plot.get_figure()
    fig.savefig(f"{save_directory}/{filename}.pdf")
    #plt.show()
    plt.clf()

if __name__ == "__main__":
    main()
