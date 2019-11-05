"""
This file contains helper code for plotting
"""
from __future__ import unicode_literals
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import auc
matplotlib.use('Agg')

#import matplotlib as mpl
#mpl.use('module://backend_interagg')
#https://www.adressa.no/pluss/okonomi/2018/09/08/Prora-eierne-g%C3%A5r-for-frifinnelse-i-ankesaken-17471354.ece
import pandas as pd
import matplotlib.pyplot as plt
# plt.rc('font', family='serif')
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

plt.rc('text', usetex=True)
import seaborn as sns
import numpy as np
import json
from utils.runutils import rundict
sim_nn_results ={
"bal": 0.093,
"bal_std": 0.043,
"car": 0.072,
"car_std": 0.010,
"cmc": 0.501,
"cmc_std": 0.026,
"eco": 0.179,
"eco_std": 0.085,
"glass": 0.162,
"glass_std": 0.049,
"hay": 0.254,
"hay_std": 0.084,
"heart": 0.211,
"heart_std": 0.084,
"iris": 0.033,
"iris_std": 0.024,
"mam": 0.207,
"mam_std": 0.027,
"mon": 0.188,
"mon_std": 0.119,
"pim": 0.273,
"pim_std": 0.027,
"ttt": 0.024,
"ttt_std": 0.016,
"use": 0.067,
"use_std": 0.033,
"who": 0.282,
"who_std": 0.034,
"bupa": 1.32,
"bupa_std": 0.66,
"housing": 2.13,
"housing_std": 0.31,
"machine": 35.2,
"machine_std": 6.5,
"servo": 0.271,
"servo_std": 0.063,
"yacht": 0.055,
"yacht_std": 0.059}

def plotTraining(folded_results,method,loss,save_directory):
    fold_counter = 0
    datadict = {"epochs": np.arange(0, len(folded_results[0][method]["history"]["loss"])).tolist()}
    collist = []
    for fold in folded_results:
        results = fold[method]
        history = results["history"]
        collist.append(f"fold_{fold_counter}")
        if loss:
            datadict[f"fold_{fold_counter}"] = history["loss"]
        fold_counter += 1
    df = pd.DataFrame(datadict ,columns=["epochs"]+collist)
    df = df.set_index(df.epochs)
    columns = [df[col] for col in collist]
    sns_plot = sns.tsplot(columns, color="indianred")
    fig = sns_plot.get_figure()
    if loss:
        fig.savefig(f"{save_directory}/{method}_loss.pdf")
    else:
        fig.savefig(f"{save_directory}/{method}_acc.pdf")
    plt.clf()


def plotResults(datasets, results, save_directory, k_folds):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    x_axis = {"datasets": datasets}
    y_series = dict()
    methods = results[datasets[0]][0].keys()
    for method in methods:
        y_series[method] = dict()
        y_series[method]["datasets"] = datasets
        for i in range(0, k_folds):
            y_series[method][f"{method}_{i}fold"] = list()
    for dataset in datasets:
        folded_results = results[dataset]
        fold_counter = 0
        for fold in folded_results:
            for method in fold:
                res_loss = fold[method]["res_loss"]
                y_series[method][f"{method}_{fold_counter}fold"].append(res_loss)
            fold_counter += 1
            #collist.append(f"fold_{fold_counter}")
    fig, ax = plt.subplots()
    for method in methods:
        collist = [f"{method}_{fold}fold" for fold in range(0, k_folds)]
        df = pd.DataFrame(y_series[method], columns=["datasets"]+collist)
        df = df.set_index(df.datasets)
        columns = [df[col] for col in collist]
        sns_plot = sns.tsplot(columns, ax=ax)
        fig = sns_plot.get_figure()
    fig.savefig(f"{save_directory}/all_methods_loss.pdf")


def plotResults2(datasets, results, save_directory, k_folds):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    dataList = list()
    for dataset in datasets:
        folded_results = results[dataset]
        fold_counter = 0
        for fold in folded_results:
            for method in fold:
                res_loss = fold[method]["ret_loss"]
                dict = {"dataset": dataset, "method": method,
                        "fold": fold_counter, "value": res_loss}
                dataList.append(dict)
            fold_counter += 1
    df = pd.DataFrame(dataList)
    fig, ax = plt.subplots()
    sns.set(font_scale=0.4)
    sns_plot = sns.lineplot(x="dataset", y="value",
                            style="method", hue="method", data=df)
    ax.set(xlabel='common xlabel', ylabel='common ylabel')
    fig = sns_plot.get_figure()
    fig.savefig(f"{save_directory}/all_methods_loss.pdf")
    plt.clf()


def plotResults3(datasets, results, save_directory, k_folds):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    dataList = list()
    for dataset in datasets:
        dataset_result = results[dataset]
        fold_counter = 0
        for key in dataset_result.keys():
            fold = dataset_result[key]
            for method in fold:
                res_loss = fold[method]["ret_loss"]
                dict = {"dataset": dataset, "method": method,
                        "fold": fold_counter, "value": res_loss}
                dataList.append(dict)
            fold_counter += 1
    df = pd.DataFrame(dataList)
    grp = df.groupby(["dataset"],as_index=False)
    df["dataset_mean"] = grp.value.transform("mean")
    sorted_df = df.sort_values(by=["dataset_mean", "method"])
    fig, ax = plt.subplots()
    sns.set(font_scale=0.4)
    sns_plot = sns.lineplot(x="dataset", y="value",
                            hue="method", data=sorted_df)
    ax.set(xlabel='dataset', ylabel='retrieval loss')
    fig = sns_plot.get_figure()
    fig.savefig(f"{save_directory}/all_methods_loss.pdf")
    plt.clf()


def plotNResults(datasets, nresults, save_directory, k_folds, n, removeoptimizer = False):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)

    folds = [i for i in range(0, k_folds)]
    lastdf = None
    for i in range(0, len(nresults)):
        dataList = list()
        results = nresults[i]
        for dataset in datasets:
            dataset_result = results[dataset]
            fold_counter = 0
            for key in dataset_result.keys():
                fold = dataset_result[key]
                for method in fold:
                    if removeoptimizer:
                        methodname = method.split(":")[0]
                    else:
                        methodname = method
                    res_loss = fold[method]["ret_loss"]
                    dict = {"dataset": dataset, "method": methodname,
                            "fold": fold_counter, "value": res_loss, "i": i} #TODO: is this value: correct?
                    dataList.append(dict)
                fold_counter += 1
        df = pd.DataFrame(dataList)
        grp = df.groupby(["dataset"],as_index=False)
        heh = grp.value.transform("mean")
        #df["dataset_mean"] = grp.value.transform("mean")
        if lastdf is None:
            lastdf = df
        else:
            lastdf = pd.concat((lastdf, df))
    by_row_index = lastdf.groupby(["dataset","method"],as_index=False)
    df_means = by_row_index.mean()
    sorted_df = df_means.sort_values(by=["value","method"])
    with sns.plotting_context("poster",font_scale=0.8, rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        sns_plot = sns.lineplot(x="dataset", y="value",
                                hue="method",
                                data=sorted_df)
        ax.set(xlabel='dataset', ylabel='retrieval loss')
        fig = sns_plot.get_figure()
        fig.savefig(f"{save_directory}/all_methods_loss.pdf")
        plt.clf()


def plotTrainings(datasets, results, save_directory, k_folds):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    dataList = list()
    for dataset in datasets:
        dataset_result = results[dataset]
        fold_counter = 0
        for foldnumber in dataset_result.keys():
            fold = dataset_result[foldnumber]
            for method in fold:
                training_loss = fold[method]["training_loss_hist"]
                epochnum = 0
                for epoch in training_loss:
                    epochnum += 1
                    thisdict = {"epoch": epochnum, "method": method,
                                "fold": fold_counter, "value": epoch}
                    dataList.append(thisdict)
            fold_counter += 1
        df = pd.DataFrame(dataList)
        setLateXFonts()
        fig, ax = plt.subplots()
        sns.set(font_scale=0.4)
        sns_plot = sns.lineplot(x="epoch", y="value",
                                style="method", hue="method", data=df)
        ax.set(xlabel='dataset', ylabel='retrieval loss')
        fig = sns_plot.get_figure()
        fig.savefig(f"{save_directory}/{dataset}_trainingloss.pdf")
        plt.clf()

def strIsFloat(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def plotNTrainings(nresults, save_directory, filename, font_scale,
                   removeoptimizer = False, showevals = False,
                   methods = None, datasetsize=1, hue_order=None,
                   filter_methods=None, pdf = True, legend_loc=1):
    from utils.runutils import maketrainingdata_dict
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    
    if filter_methods is None:
        filter_methods = []
    lastdf = None
    xkey = "epochs"
    maketrainingdatakey = "split"
    if showevals:
        xkey = "evals"
    for i in range(0,len(nresults)):
        dataList = list()
        results = nresults[i]
        for dataset in results.keys():
            dataset_result = results[dataset]
            fold_counter = 0
            for foldnumber in dataset_result.keys():
                fold = dataset_result[foldnumber]
                for method in fold:
                    if method in filter_methods:
                        continue
                    evalFunc = lambda x: 1

                    methodkey = method.split(":")[0]

                    if showevals:
                        evalFunc = methods[methodkey]["maketrainingdata"]["factorfunc"]


                    if removeoptimizer:
                        methodname = method.split(":")[0]
                    else:
                        methodname = method
                        if strIsFloat(methodname):
                            methodname = "alpha-"+methodname
                    training_loss = fold[method]["training_retain_loss_hist"]
                    epochnum = 0
                    for epochvalue in training_loss:
                        epochnum += 1
                        thisdict = {xkey: epochnum*evalFunc(datasetsize), "method": methodname,
                        "fold": fold_counter, "value": epochvalue, "n": i}
                        dataList.append(thisdict)
                fold_counter += 1
        df = pd.DataFrame(dataList)
        grp = df.groupby(["method", xkey],as_index=False)
        df["mean_value"] = grp.value.transform("mean")
        if lastdf is None:
            lastdf = df
        else:
            lastdf = pd.concat((lastdf, df),ignore_index=True)
        #df["dataset_mean"] = grp.value.transform("mean")
    #test = lastdf.loc[(lastdf['epoch'] == 500) & (lastdf['method'] == "dees:rprop")]
    lastdf = lastdf.drop(columns="value")
    lastdf = lastdf.drop_duplicates(subset=["method", xkey, "n", "mean_value"],
                                    keep='first')
    
    #sns.set(font_scale=0.7)
    setLateXFonts()
    with sns.plotting_context("poster", font_scale=font_scale,
                              rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        if hue_order is None:
            sns_plot = sns.lineplot(x=xkey, y="mean_value",
                                    style="method", hue="method", data=lastdf)
        else:
            sns_plot = sns.lineplot(x=xkey, y="mean_value",
                                    hue="method",
                                    hue_order=hue_order,data=lastdf)

        ax.set(xlabel=xkey, ylabel='retrieval loss')
        ax.legend(loc=legend_loc)
        # ax.set_xlim(0,20)
        if showevals:
            ax.set(xscale="log")
            ax.set(xlabel="datapoints evaluated")
        #plt.setp(ax.get_legend().get_texts(), fontsize='20')
        #plt.setp(ax.get_legend().get_title(), fontsize='20')
        fig = sns_plot.get_figure()
        if pdf is True:
            fig.savefig(f"{save_directory}/{filename}.pdf",bbox_inches='tight')
        else:
            fig.savefig(f"{save_directory}/{filename}.eps",bbox_inches='tight')
        plt.clf()



def create_tpr_fpr(path):
    test_df = pd.read_csv(path)
    test_data = []
    for group in test_df.groupby("class"):
        test_data.append(group[1].iloc[0:100])
    df = pd.concat(test_data)
    distances = pdist(df.drop("class", axis=1), metric='euclidean')
    dist_matrix = squareform(distances)

    same_fish = pdist(df, metric=lambda u, v: u[-1] == v[-1])
    same_fish_matrix = squareform(same_fish)
    same_fish_matrix = same_fish_matrix + np.eye(same_fish_matrix.shape[0])

    n_same = np.count_nonzero(same_fish_matrix)
    n_diff = np.size(same_fish_matrix) - np.count_nonzero(same_fish_matrix)

    tprs = []
    fprs = []

    for t in np.arange(0.0, 2.02, 0.02):
        predict_is_same = dist_matrix <= t
        true_positive_mx = np.logical_and(predict_is_same, same_fish_matrix)
        n_true_positive = np.count_nonzero(true_positive_mx)
        false_positive_mx = np.logical_and(predict_is_same, np.logical_not(same_fish_matrix))
        n_false_positive = np.count_nonzero(false_positive_mx)
        tprs.append(n_true_positive / n_same)
        fprs.append(n_false_positive / n_diff)
    return tprs, fprs

def plotEmbeddingClusters(save_directory, trainedmodel, randommodel,
                          filename, font_scale,
                          allfeatures, alltargets,
                          indexes, classes, maxdatapoints,
                          removeoptimizer=False,
                          methods=None, hue_order=None,
                          pdf=True):
    setLateXFonts()
    #reduce data to the indexes given by stratified kfold
    onehottargets = alltargets[indexes]
    features = allfeatures[indexes]
    #convert the onehot to numbers for easier plotting
    targets = np.asarray([np.where(r==1)[0][0] for r in onehottargets])

    #extracting the indexes in datasubset that contains the target classes specified
    classindexes = np.asarray([],dtype=int)
    for c in classes:
        tmp = np.where(targets == c)
        classindexes = np.append(classindexes, tmp)

    # reduce the number of datapoint to something easier to plot
    maxnum = len(classindexes) \
        if len(classindexes) < maxdatapoints else maxdatapoints
    np.random.shuffle(classindexes)
    classindexes = classindexes[:maxnum]

    modelinput = features[classindexes]
    modeltargets = targets[classindexes]

    output = randommodel.predict(modelinput)

    #
    # calculate and plot the PCA and T-SNE random embeddings
    #

    #PCA
    pca = PCA(n_components=2)
    pca.fit(output)
    output_pca = pca.transform(output)

    pc_df = pd.DataFrame(data=output_pca, columns=['PC1', 'PC2'])
    pc_df["Cluster"] = modeltargets
    print(f"pca explained variance: {pca.explained_variance_ratio_}")
    with sns.plotting_context("poster", font_scale=font_scale,
                              rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        sns_plot = sns.lmplot(x="PC1", y="PC2",
                              data=pc_df,
                              fit_reg=False,
                              hue='Cluster',  # color by cluster
                              legend=True,
                              scatter_kws={"s": 80})
        if pdf is True:
            sns_plot.savefig(f"{save_directory}/{filename}-pca-random.pdf",bbox_inches='tight')
        else:
            sns_plot.savefig(f"{save_directory}/{filename}-pca-random.eps",bbox_inches='tight')
        plt.clf()

    #T-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    output_tsne = tsne.fit_transform(output)

    pc_df = pd.DataFrame(data=output_tsne, columns=['tsne-one', 'tsne-two'])
    pc_df["Cluster"] = modeltargets
    with sns.plotting_context("poster", font_scale=font_scale,
                              rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        sns_plot = sns.lmplot(x="tsne-one", y="tsne-two",
                              data=pc_df,
                              fit_reg=False,
                              hue='Cluster',  # color by cluster
                              legend=True,
                              scatter_kws={"s": 80})
        if pdf is True:
            sns_plot.savefig(f"{save_directory}/{filename}-tsne-random.pdf",bbox_inches='tight')
        else:
            sns_plot.savefig(f"{save_directory}/{filename}-tsne-random.eps",bbox_inches='tight')
        plt.clf()


    # compute the embeddings with trained model
    output = trainedmodel.predict(modelinput)

    #
    # calculate and plot the PCA and T-SNE trained embeddings
    #

    # PCA
    pca = PCA(n_components=2)
    pca.fit(output)
    output_pca = pca.transform(output)
    pc_df = pd.DataFrame(data = output_pca, columns = ['PC1', 'PC2'])
    pc_df["Cluster"] = modeltargets
    print(f"pca explained variance: {pca.explained_variance_ratio_}")
    with sns.plotting_context("poster", font_scale=font_scale,
                              rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        sns_plot = sns.lmplot(x="PC1", y="PC2",
                              data=pc_df,
                              fit_reg=False,
                              hue='Cluster',  # color by cluster
                              legend=True,
                              scatter_kws={"s": 80})
        if pdf is True:
            sns_plot.savefig(f"{save_directory}/{filename}-pca-trained.pdf",bbox_inches='tight')
        else:
            sns_plot.savefig(f"{save_directory}/{filename}-pca-trained.eps",bbox_inches='tight')
        plt.clf()

    # T-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    output_tsne = tsne.fit_transform(output)

    pc_df = pd.DataFrame(data=output_tsne, columns=['tsne-one', 'tsne-two'])
    pc_df["Cluster"] = modeltargets
    with sns.plotting_context("poster", font_scale=font_scale,
                              rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        sns_plot = sns.lmplot(x="tsne-one", y="tsne-two",
                              data=pc_df,
                              fit_reg=False,
                              hue='Cluster',  # color by cluster
                              legend=True,
                              scatter_kws={"s": 80})
        if pdf is True:
            sns_plot.savefig(f"{save_directory}/{filename}-tsne-trained.pdf",bbox_inches='tight')
        else:
            sns_plot.savefig(f"{save_directory}/{filename}-tsne-trained.eps",bbox_inches='tight')
        plt.clf()
    return classindexes, modeltargets

def plotNRealTrainings(nresults, save_directory, filename, font_scale,
                       removeoptimizer=False, showevals=False,
                       methods=None, datasetsize=1, hue_order=None,
                       pdf=True):
    from utils.runutils import maketrainingdata_dict
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    lastdf = None
    xkey = "epochs"
    maketrainingdatakey = "split"
    if showevals:
        xkey = "evals"
    for i in range(0,len(nresults)):
        dataList = list()
        results = nresults[i]
        for dataset in results.keys():
            dataset_result = results[dataset]
            fold_counter = 0
            for foldnumber in dataset_result.keys():
                fold = dataset_result[foldnumber]
                for method in fold:
                    evalFunc = lambda x: 1

                    methodkey = method.split(":")[0]

                    if showevals:
                        evalFunc = methods[methodkey]["maketrainingdata"]["factorfunc"]


                    if removeoptimizer:
                        methodname = method.split(":")[0]
                    else:
                        methodname = method
                        if strIsFloat(methodname):
                            methodname = "alpha-"+methodname
                    training_loss = fold[method]["training_losses"]
                    epochnum = 0
                    for epochvalue in training_loss:
                        epochnum += 1
                        thisdict = {xkey: epochnum*evalFunc(datasetsize), "method": methodname,
                        "fold": fold_counter, "value": epochvalue, "n": i}
                        dataList.append(thisdict)
                fold_counter += 1
        df = pd.DataFrame(dataList)
        grp = df.groupby(["method", xkey],as_index=False)
        df["mean_value"] = grp.value.transform("mean")
        if lastdf is None:
            lastdf = df
        else:
            lastdf = pd.concat((lastdf, df),ignore_index=True)
        #df["dataset_mean"] = grp.value.transform("mean")
    #test = lastdf.loc[(lastdf['epoch'] == 500) & (lastdf['method'] == "dees:rprop")]
    lastdf = lastdf.drop(columns="value")
    lastdf = lastdf.drop_duplicates(subset=["method", xkey, "n", "mean_value"],
                                    keep='first')
    #sns.set(font_scale=0.7)
    setLateXFonts()
    with sns.plotting_context("poster", font_scale=font_scale,
                              rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        if hue_order is None:
            sns_plot = sns.lineplot(x=xkey, y="mean_value",
                                    hue="method", data=lastdf)
        else:
            sns_plot = sns.lineplot(x=xkey, y="mean_value",
                                    hue="method",
                                    hue_order=hue_order,data=lastdf)
        ax.set(xlabel=xkey, ylabel='training loss')
        if showevals:
            ax.set(xscale="log")
            ax.set(xlabel="datapoints evaluated")
        #plt.setp(ax.get_legend().get_texts(), fontsize='20')
        #plt.setp(ax.get_legend().get_title(), fontsize='20')
        fig = sns_plot.get_figure()
        if pdf is True:
            fig.savefig(f"{save_directory}/{filename}.pdf",bbox_inches='tight')
        else:
            fig.savefig(f"{save_directory}/{filename}.eps",bbox_inches='tight')
        plt.clf()

from matplotlib.ticker import FormatStrFormatter

def plotNAlphaResults(nresults, save_directory, font_scale=0.8,
                      file_name="all_methods_loss.pdf", drawdev = False,
                      removeoptimizer = False, pdf = True):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)

    #folds = [i for i in range(0, k_folds)]
    lastdf = None
    for i in range(0,len(nresults)):
        dataList = list()#testing
        results = nresults[i]
        for dataset in results.keys():
            dataset_result = results[dataset]
            fold_counter = 0
            for key in dataset_result.keys():
                fold = dataset_result[key]
                for method in fold:
                    if removeoptimizer:
                        methodname = method.split(":")[0]
                    else:
                        methodname = method
                    res_loss = fold[method]["ret_loss"]
                    #method = round(float(methodname), 2)
                    thisdict = {"dataset": dataset, "method": method,
                            "fold": fold_counter, "value": res_loss, "i": i}
                    dataList.append(thisdict)
                fold_counter += 1
        df = pd.DataFrame(dataList)
        grp = df.groupby(["dataset","method"], as_index=False)
        df["value"] = grp.value.transform("mean")
        #df["dataset_mean"] = grp.value.transform("mean")
        if lastdf is None:
            lastdf = df
        else:
            lastdf = pd.concat((lastdf, df),ignore_index=True)
    #lastdf = lastdf.drop(columns="value")
    lastdf = lastdf.drop_duplicates(subset=["method","i","value"],keep='first')
    #by_row_index = lastdf.groupby(["dataset", "method"], as_index=False)
    grp_method = df.groupby(["method", "dataset"], as_index=False)
    lastdf["method_mean"] = grp_method.value.transform("mean")
    #df_means = by_row_index.mean()
    #sorted_df = df_means.sort_values(by=["method"])
    with sns.plotting_context("poster",font_scale=font_scale, rc={"lines.linewidth": 1}):
        fig, ax = plt.subplots()
        if drawdev:
            sns_plot = sns.lineplot(x="method", y="value",
                                    data=lastdf, sort=False)
        else:
            sns_plot = sns.lineplot(x="method", y="method_mean",
                                    data=lastdf, sort=False)
        #sns_plot = sns.lineplot(x="method", y="value", data=lastdf)
        ax.set(xlabel=r'$\alpha$', ylabel='retrieval loss')
        divider = int(df.method.unique().shape[0]/5)
        #print(f"divider:{divider}")
        print(df.method[::divider])
        sns_plot.set(xticks=df.method[::divider])
        #sns_plot.set(xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8])
        ###ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        fig = sns_plot.get_figure()
        plt.tight_layout()
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        #ax.set_xticklabels([str(round(float(label), 2)) for label in labels])
        if pdf is True:
            fig.savefig(f"{save_directory}/{filename}.pdf",bbox_inches='tight')
        else:
            fig.savefig(f"{save_directory}/{filename}.eps",bbox_inches='tight')
        plt.clf()


def plotAlphaResults(datasets, results, save_directory, k_folds):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    dataList = list()
    for dataset in datasets:
        dataset_result = results[dataset]
        fold_counter = 0
        for key in dataset_result.keys():
            fold = dataset_result[key]
            for method in fold:
                res_loss = fold[method]["ret_loss"]
                dict = {"dataset": dataset, "method": method,
                        "fold": fold_counter, "value": res_loss}
                dataList.append(dict)
            fold_counter += 1
    df = pd.DataFrame(dataList)
    grp = df.groupby(["dataset"],as_index=False)
    #df["dataset_mean"] = grp.value.transform("mean")
    grp["dataset_mean"] = grp.value.transform("mean")
    sorted_df = grp.sort_values(by=["dataset_mean", "method"])
    fig, ax = plt.subplots()
    sns.set(font_scale=0.4)
    sns_plot = sns.lineplot(x="method", y="dataset_mean", style="dataset",
                            hue="dataset", data=sorted_df)
    ax.set(xlabel='dataset', ylabel='retrieval loss')
    fig = sns_plot.get_figure()
    fig.savefig(f"{save_directory}/all_methods_loss.pdf")
    plt.clf()


def readJson(resultdir):
    #run = "gabelstop-allrun-correct2018_09_24_21_55_55"

    jsonFile = f"{resultdir}/data.json"
    output_json = json.load(open(jsonFile))
    return output_json

def jsonToTableDF(output_json):
    methods = dict()
    for datasetname in output_json.keys():
        for fold_number in output_json[datasetname].keys():
            fold = output_json[datasetname][fold_number]
            for method_name in fold.keys():

                if method_name in methods:
                    method = methods[method_name]
                else:
                    method = dict()
                    methods[method_name] = method

                if datasetname in method:
                    method[datasetname].append(fold[method_name]["ret_loss"])
                else:
                    resultlist = list()
                    resultlist.append(fold[method_name]["ret_loss"])
                    method[datasetname] = resultlist
    #print(methods.keys())

    methodlist = list()
    yadict = dict()
    for method_name in methods.keys():
        yadict[method_name] = dict()
        for datasetname in methods[method_name].keys():
            templist = methods[method_name][datasetname]
            loss = np.sum(templist)/len(templist)
            std = np.std(templist)
            yadict[method_name][datasetname] = loss
          #yadict[method_name][f"{datasetname}_std"] = std
        yadict[method_name]["method_name"] = method_name
        methodlist.append(yadict[method_name])

    df = pd.DataFrame(data=methodlist)
    df = df.set_index("method_name")
    return df


def getGabelTableDF(cols):
    newdict = dict()
    for key in sim_nn_results.keys():
        if key in cols and "std" not in key:
            newdict[key] = sim_nn_results[key]
            newdict["method_name"] = "gabel_orig"
    sim_nn_df = pd.DataFrame(data=newdict, columns=newdict.keys(), index=[0])
    sim_nn_df = sim_nn_df.set_index("method_name")
    return sim_nn_df


def nJsonToDF(nresults, k_folds, removeoptimizer=False):
    folds = [i for i in range(0,k_folds)]
    lastdf = None
    for i in range(0, len(nresults)):
        dataList = list()
        results = nresults[i]
        stats = dict()
        for dataset in results.keys():
            dataset_result = results[dataset]
            fold_counter = 0
            for key in dataset_result.keys():
                fold = dataset_result[key]
                for method in fold:
                    if removeoptimizer:
                        methodname = method.split(":")[0]
                    else:
                        methodname = method
                    res_loss = fold[method]["ret_loss"]
                    tdict = {"dataset": dataset, "method": methodname,
                             "fold": fold_counter, "value": res_loss,
                             "n": i}
                    dataList.append(tdict)
                fold_counter += 1
        df = pd.DataFrame(dataList)
        grp = df.groupby(["dataset","method"],as_index=False)
        #transformed_grp = grp.value.transform("mean")
        df["value"] = grp.value.transform("mean")
        if lastdf is None:
            lastdf = df
        else:
            lastdf = pd.concat((lastdf, df),ignore_index=True)
    return lastdf.drop_duplicates(subset=["dataset","method","n","value"]).drop(columns=["fold"])


def jsonToDF(results):
    dataList = list()
    stats = dict()
    for dataset in results.keys():
        dataset_result = results[dataset]
        fold_counter = 0
        for key in dataset_result.keys():
            fold = dataset_result[key]
            for method in fold:
                res_loss = fold[method]["ret_loss"]
                tdict = {"dataset": dataset, "method": method,
                         "fold": fold_counter, "value": res_loss}
                dataList.append(tdict)
                # templist.append(dict)
            #datalist.extend(templist)
            fold_counter += 1
    df = pd.DataFrame(dataList)
    return df

def setLateXFonts():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

def plotDf(df, save_directory, filename, drawdev=False, 
           font_scale=0.8, hue_order=None, legend_loc=1):
    # need df with row: <dataset,method,fold,value> then
    # sns.lineplot(data=df,x="dataset",y="value",style="method)
    sns.set_style("whitegrid")
    setLateXFonts()
    with sns.plotting_context("poster",font_scale=font_scale, rc={"lines.linewidth": font_scale, "grid.linewidth": font_scale}):
        fig, ax = plt.subplots()

        if drawdev:
            _y = "value"
        else:
            _y="method_mean"

        if hue_order is None:
            sns_plot = sns.lineplot(x="dataset", y=_y,
                                    hue="method",
                                    data=df, sort=False)
        else:
            sns_plot = sns.lineplot(x="dataset", y=_y,
                                    hue="method", hue_order=hue_order,
                                    data=df, sort=False)
        ax.set(xlabel='dataset', ylabel='retrieval loss')
        ax.legend(loc=legend_loc)
        fig = sns_plot.get_figure()
        fig.savefig(f"{save_directory}/{filename}.pdf",bbox_inches='tight')
        plt.show()
        plt.clf()

def generateGabelDF(k_folds, datasets):
    gabelJson = sim_nn_results
    #datasets = [key for key in gabelJson.keys() if "std" not in key]
    datalist = list()
    for key in datasets:
        std_key = key+"_std"
        desired_mean = float(gabelJson[key])
        desired_std_dev = float(gabelJson[std_key])
        data = generateData(desired_mean, desired_std_dev, k_folds)
        fold_counter = 0 
        for datapoint in data:
            datalist.append({"dataset": key, "method": "gabel",
                             "fold": fold_counter, "value": datapoint})
            fold_counter += 1
    return datalist


def generateData(desired_mean, desired_std_dev, num_samples):
    samples = np.random.normal(loc=0.0, scale=desired_std_dev,
                               size=num_samples)

    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}"
          .format(actual_mean, actual_std))

    zero_mean_samples = samples - (actual_mean)

    zero_mean_mean = np.mean(zero_mean_samples)
    zero_mean_std = np.std(zero_mean_samples)
    print("True zero samples stats : mean = {:.4f} stdv = {:.4f}"
          .format(zero_mean_mean, zero_mean_std))

    scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)
    scaled_mean = np.mean(scaled_samples)
    scaled_std = np.std(scaled_samples)
    print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}"
          .format(scaled_mean, scaled_std))

    final_samples = scaled_samples + desired_mean
    final_mean = np.mean(final_samples)
    final_std = np.std(final_samples)
    print("Final samples stats     : mean = {:.4f} stdv = {:.4f}"
          .format(final_mean, final_std))

    return final_samples


def plotJson(resultfile, k_folds, savedir, filename,
             add_gabel=False, drawdev=False, plot_loc=1):
    json_data = readJson(resultfile)
    df = jsonToDF(json_data)
    if add_gabel:
        gdf = generateGabelDF(k_folds, df.dataset.unique())
        df = df.append(gdf)
    grp_ds = df.groupby(["dataset"], as_index=False)
    df["dataset_mean"] = grp_ds.value.transform("mean")
    grp_method = df.groupby(["method", "dataset"], as_index=False)
    df["method_mean"] = grp_method.value.transform("mean")
    sorted_df = df.sort_values(by=["dataset_mean", "method"])
    print(df.head(10))
    plotDf(sorted_df, savedir, filename, drawdev, plot_loc)


def plotNJson(resultfile, k_folds, savedir, filename, sortmethodname,
              add_gabel=False, drawdev=False, font_scale=0.8, removeoptimizer=False,
              hue_order=None, dataset_order=None, legend_loc=1):
    json_data = readJson(resultfile)
    df = nJsonToDF(json_data, k_folds, removeoptimizer)
    n = 5
    if add_gabel:
        for i in range(0,n):
            gdf = generateGabelDF(k_folds, df.dataset.unique())
            df = df.append(gdf)
    grp_ds = df.groupby(["dataset"], as_index=False)
    df["dataset_mean"] = grp_ds.value.transform("mean")
    grp_method = df.groupby(["method", "dataset"], as_index=False)
    df["method_mean"] = grp_method.value.transform("mean")
    df["df"] = df.apply(lambda row: bs(row, df, sortmethodname), axis=1)
    df['sort_method_value'] = df \
        .apply(lambda row: getCorrespondingMethodValue(row, df, sortmethodname),
               axis=1)
    if dataset_order is None:
        sorted_df = df.sort_values(by=["sort_method_value", "method"])
    else:
        df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order)
        sorted_df = df.sort_values(by=["dataset"])
        print("now we are here weee")

    sorted_df["method"] = sorted_df.apply(lambda row: rundict[row["method"]]["figname"])
    plotDf(sorted_df, savedir, filename, drawdev,
           font_scale, hue_order, legend_loc)


def bs(row,df,sortmethodname):
    return 1


def getCorrespondingMethodValue(row, df, method):
    if method in row['method']:
        return row['method_mean']
    else:
        return df.loc[(df['dataset'] == row['dataset'])
                      & (method == df['method'])
                      & (row["n"] == df["n"])]['method_mean'].iloc[0]
