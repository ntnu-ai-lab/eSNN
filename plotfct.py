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
import argparse

# public double getValue(double arg0) {
# 	try {
# 		Float value1 = Double.valueOf(fct.getMin()).floatValue();
# 		Float value2 = Double.valueOf(Math.abs(arg0) + fct.getMin()).floatValue();
# 		if(arg0 > 0) {
# 			return fct.calculateSimilarity(value1, value2).getValue();
# 		} else {
# 			return fct.calculateSimilarity(value2, value1).getValue();
# 		}
# 	} catch (Exception e) {
# 		return 0;
# 	}
# }
my_min = 0.0

def getValue(arg0,param,maxv,minv):
    val1 = my_min
    val2 = abs(arg0)+my_min
    if arg0 > 0:
        return sim(val1,val2,param,maxv,minv)
    else:
        return sim(val2,val1,param,maxv,minv)

def sim(q,c,paramL,maxv,minv):
    d = q-c
    maxrange = maxv-minv # max-min
    if d < 0:
        return f(d,paramL,maxrange)
    else:
        return f(d,paramL,-maxrange)

def f(value,exponent,diff):
    return myfilter(np.power(value/diff + 1.0, exponent))

def mySimFct(c1,c2,param,diff):
    return

def myrounder(num):
    return round(num*100,0)/100.0
def myfilter(num):
    if num < 0.0 or num > 1:
        return -1
    else:
        return num



#t1 = np.arange(-1.0, 1.0, .002)

def plot(x,param,maxv,minv):
    #plt.figure(1)
    #plt.subplot(211)
    y = [getValue(xi,param,maxv,minv) for xi in x]
    #print(y)
    return sns.lineplot(x=x, y=y)
    #plt.show()

import sys
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import scipy.integrate as integrate

def main():
    parser = argparse.ArgumentParser(description='train NN classification'
                                     + ' model on datasets against CBR!')
    parser.add_argument('--filename', metavar='filename', type=str,
                        help='Filename to save the figure to', required=True)
    parser.add_argument('--font_scale', metavar='font_scale', type=float,
                        help='UI scale for the figure.', required=True)
    parser.add_argument('--dataset', metavar='dataset', type=str,
                        help='Dataset to draw boxplot from', required=True)
    parser.add_argument('--column', metavar='column', type=str,
                        help='Dataset column to draw boxplot from')
    args = parser.parse_args()
    d = Dataset(args.dataset)
    dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, False, n_splits=5)

    df = d.df[d.df.columns.difference(dsl.getNonNumericalCols())].apply(pd.to_numeric)
    
    #df = dsl.df

    if "dataset" in args and args.column is None:
        print(list(df))
        sys.exit(0)

    values = df[args.column].values
    q1 = np.percentile(values,25)
    q3 = np.percentile(values,75)
    param = abs(q1-q3)
    #param = 4
    maxv =  np.amax(values)
    minv =  np.amin(values)
    diff = maxv-minv
    t1 = np.arange(-diff, diff, (2*diff)/100)
    result = integrate.quad(lambda x: getValue(x, param, maxv, minv), -diff, diff)
    result2 = integrate.quad(lambda x: getValue(x, param, maxv, minv), -param, param)
    print(f"q1: {q1} q3: {q3} param: {param}")
    print(f"diff: {diff} maxv: {maxv} minv: {minv}")
    print(f"integrated: {result[0]}")
    print(f"integrated between -diff and +diff: {result2[0]}")
    sns.set_style("whitegrid")
    with sns.plotting_context("poster",font_scale=args.font_scale,
                              rc={"lines.linewidth": args.font_scale,
                                  "grid.linewidth": args.font_scale}):

        ax = plot(t1, param,maxv,minv)
        fig = ax.get_figure()
        fig.add_subplot(111)
        ax2 = ax.twiny()
        fig.subplots_adjust(bottom=0.2)
        lines = [-param,param]
        ax2.set_xticks(lines)
        ax2.set_xlim(ax.get_xlim())
        for line in lines:
            ax.axvline(x=line, color='k', linestyle='--')
            print(f"printing line at x={line}")
        plt.savefig(args.filename)
def parametrizedIntegralOfCurve(fromx,tox):
    def integralFunc(param):
        return integrate.quad(lambda x: getValue(x, param, maxv, minv), -diff, diff)

def fitcurve(fromx,tox):
    xdata = np.arange(fromx,tox,1.0/100.0)
    # ydata = 
    func = parmetrizedIntegralOfCurve(fromx,tox)
    param_opt, cov = scipy.optimize.curve_fit(func,xdata=xdata,ydata=ydata,p0=[1])
    return n

if __name__ == "__main__":
    main()
