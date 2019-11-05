from dataset.dataset import *
import pandas as pd
from sklearn import datasets
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer,\
    LabelBinarizer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from dataset.binarizer import MyLabelBinarizer
from sklearn_pandas import DataFrameMapper, cross_val_score
#from sklearn.datasets import Bunch
import math
from sklearn.utils import Bunch
import numpy as np

__name__ = "dataset_to_sklearn"


class SKLearnDataset():

    def __init__(self, data, featurecolsfrom, featurecolsto, targetcolsfrom,
                 targetcolsto, colnames, target_names, dataset, isregression,
                 targetcolnames):
        self.multiclass = False
        self.datacontent = data
        self.featurecolsfrom = featurecolsfrom
        self.featurecolsto = featurecolsto
        self.targetcolsfrom = targetcolsfrom
        self.targetcolsto = targetcolsto
        self.colnames = colnames
        self.target_names = target_names
        self.targetcolnames = targetcolnames
        self.dataset = dataset
        self.isregression = isregression
        self.df = pd.DataFrame(data=self.datacontent, columns=self.colnames)

    #def __getattribute__(self, key):
    #    return object.myGetAttribute(key)

    def containsNonNumericalCols(self):
        for col in self.dataset.datasetInfo["cols"]:
            if col["type"] is "nominal" or col["type"] is "ordinal":
                return True
        return False

    def getNonNumericalCols(self):
        ret = []
        for col in self.dataset.datasetInfo["cols"]:
            if col["type"] is "nominal" or col["type"] is "ordinal":
                ret.append(col["name"])
        return ret

    def myGetAttribute(self, key):
        if key is "target":
            return self.getTargets()
        elif key is "data":
            return self.getFeatures()
        else:
            return object.__dict__(self, key)

    def __getattr__(self, key):
        return self.myGetAttribute(key)

    def getFeatures(self):
        return self.datacontent[:, self.featurecolsfrom:self.featurecolsto+1]

    def getTargets(self):
        return self.datacontent[:, self.targetcolsfrom:self.targetcolsto]

    def getDataFrame(self):
        return self.df

    def setIDs(self, idList):
        self.df["id"] = idList
        self.df = self.df.set_index("id")

    def getTrainingData(self, idList):
        subset = self.df.loc[idList,:]
        values = subset.values
        features = values[:, self.featurecolsfrom:self.featurecolsto+1]
        targets = values[:, self.targetcolsfrom:self.targetcolsto]
        return features, targets






def makeembeddings(categorical_vars, df):
    """
    From https://medium.com/@satnalikamayank12/on-learning-embeddings-for-categorical-data-using-keras-165ff2773fc9
    """
    for categorical_var in categorical_vars:
        model = Sequential()
        no_of_unique_cat = df_train[categorical_var].nunique()
        embedding_size = min(np.ceil((no_of_unique_cat) / 2), 50)
        embedding_size = int(embedding_size)
        vocab = no_of_unique_cat + 1
        model.add(Embedding(vocab, embedding_size, input_length=1))
        model.add(Reshape(target_shape=(embedding_size,)))
        models.append(model)

def makeFakeYield(featurearr, targetarr):
    kfold = StratifiedKFold(n_splits=5)
    stratified_fold_generator = kfold.split(featurearr, targetarr)
    train, test = next(stratified_fold_generator)
    train  = np.arange(featurearr.shape[0])
    yield train, test


def getDatasetInfo(dataset, multilabelbin=True,
                 n_splits=None, df=None, datasetInfo=None,
                 n_samples=0, n_features=0, colnames=None,
                 classcols=[], featurecols=[], targetcols=[],
                 targetcolindexes=[], dropcols=[], colmap={}
                 ):
    for col in datasetInfo["cols"]:
        if col["class"] is True:
            classcols.append(col)

    # here we need to do some checking for datasets with two classes (such as
    # wholesale customer data) In these cases we will choose one class as
    # target label in accordance with "Gabel, T., Godehardt, E.: Top-down
    # induction of similarity measures using simi- larity clouds." as this is
    # this codes original purpose, to create a similar benchmark as in that
    # paper this may be generlized later on

    if "dropcols" in datasetInfo:
        dropcols.extend(datasetInfo["dropcols"])
        for col in dropcols:
            df.drop(col, axis=1, inplace=True)
    if "dropna" in datasetInfo:
        df.dropna(inplace=True)

    if "Wholesale" in datasetInfo["dataUrl"]:
        df.pop("Channel")
        dropcols.append("Channel")

    if dropcols is not None:
        featurecols.extend([c["name"] for c in datasetInfo["cols"]
                            if c["type"] is not "skip" and c["class"] is not True
                            and c["name"] not in dropcols])
    else:
        featurecols.extend([c["name"] for c in datasetInfo["cols"]
                            if c["type"] is not "skip" and c["class"] is not True])

    df = dataset.df

    #make two separate view of the df, one for all features and one for the targetvariable
    #featuredfview = df.loc[;,featurecols]


    columns_to_scale = []
    columns_to_binarize = []
    feature_names = []
    isregression = False
    #go through all the input features and find out which type of preprocessing they need
    for i in range(0,len(datasetInfo["cols"])):
        colname = datasetInfo["cols"][i]["name"]
        coltype = datasetInfo["cols"][i]["type"]
        colclass = datasetInfo["cols"][i]["class"]
        if dropcols is not None and colname in dropcols:
            continue
        if coltype is "skip":
            continue

        values = df[colname].unique()
        foundNaN = False
        datadict = {}

        for value in values:
            if isinstance(value, (int, float, complex)) and math.isnan(value):
                foundNaN = True

        if (coltype is "nominal"
                or coltype is "ordinal")\
                and colclass is False:
            #nominal, make use of onehotencoding
            if multilabelbin:
                possible_values = []
                possible_values.extend(df[colname].unique())
                datadict["possible_values"] = [colname+"_"+v for v in possible_values]
            else:
                datadict["possible_values"] = [colname]
            datadict["type"] = "nominal"
            columns_to_binarize.append(datasetInfo["cols"][i]["name"])
            if foundNaN:
                df[colname].fillna("nan", inplace=True)
        elif (coltype is np.float32
              or coltype is np.int32)\
                and colclass is False:
            #int or float, normalize between 0 and 1
            columns_to_scale.append(datasetInfo["cols"][i]["name"])
            datadict["type"] = "number"
            if foundNaN:
                df[colname].fillna(0, inplace=True)
        #binary, do nothing
        #if target column is not set yet find it
        if colclass is True:
            targetcols.append(datasetInfo["cols"][i]["name"])
            targetcolindexes.append(i)
            if coltype is np.float32 or coltype is np.int32:
                isregression = True
        else:
            feature_names.append(colname)
        colmap[colname] = datadict

    #targetdfview = df.loc[:,targetcols]
    # make the list of different mappings for the feature columns
    featuremapperlist = []
    for col in columns_to_binarize:
        if multilabelbin:
            featuremapperlist.append((col, MyLabelBinarizer()))
        else:
            featuremapperlist.append((col, LabelEncoder()))
    for col in columns_to_scale:
        df.loc[:,(col)] = df.loc[:,(col)].apply(pd.to_numeric)
        featuremapperlist.append(([col], MinMaxScaler()))
    featuremapper = DataFrameMapper(featuremapperlist, df_out=True)

    # make the list of different mapping of the target columns

    target_names = []
    targetmapperlist = []
    for targetcolindex, targetcolname in zip(targetcolindexes, targetcols):
        datadict = {}
        targetcoltype = datasetInfo["cols"][targetcolindex]["type"]
        if targetcoltype is np.int32 or targetcoltype is np.float32:
            df.loc[:,(targetcolname)] = df.loc[:,(targetcolname)].apply(pd.to_numeric)
            targetmapperlist.append(([targetcolname], MinMaxScaler()))
            datadict["type"] = "number"
        elif targetcoltype is "nominal" or targetcoltype is "ordinal":

            datadict["type"] = "nominal"
            if multilabelbin:
                targetmapperlist.append((targetcolname, MyLabelBinarizer()))
                values = df[targetcolname].unique().tolist()
                target_names.extend(values)
                datadict["possible_values"] = [targetcolname + "_" + pv for pv in values.copy()]
            else:
                targetmapperlist.append((targetcolname, LabelEncoder()))
                datadict["possible_values"] = [targetcolname]
        colmap[targetcolname] = datadict

    targetmapper = DataFrameMapper(targetmapperlist, df_out=True)

    return featuremapper, targetmapper, target_names, \
        isregression, featurecols, targetcols, colmap

def adapt(df, featuremapper, targetmapper,
          featurecols, targetcols, datadict, colmap):

    # do the mapping through df-scikit-mapper
    mappedfeaturedf = featuremapper.fit_transform(df.loc[:, featurecols].copy())
    mappedtargetdf = targetmapper.fit_transform(df.loc[:,targetcols].copy())

    # get the mapped values as numpy arrays
    targets = mappedtargetdf.values
    features = mappedfeaturedf.values
    if "post_process" in datadict:
        features, colmap = datadict["post_process"](features, colmap)

    colnames = []
    for thiscolname, coldict in colmap.items():
        if "possible_values" in coldict:
            colnames.extend(coldict["possible_values"])
        else:
            colnames.append(thiscolname)

    return features, targets, colnames


def fromDataSetToSKLearn(dataset, multilabelbin=True, n_splits=None):
    df = dataset.df
    datasetInfo = dataset.datasetInfo
    n_samples = df.shape[0]
    n_features = df.shape[1]-datasetInfo["num_classes"]
    colnames = list(df)
    classcols = []
    #find all classes


    targetcols = []
    featurecols = []
    targetcolindexes = []
    dropcols = []
    features = None
    targets = None
    target_names = None
    isregression = False
    colmap = {}


    # get the info needed for converting the data to proper format
    if "getInfo" in datasetInfo:
        getDatasetInfo_dyn = datasetInfo["getInfo"]
        target_names, isregression, \
            featurecols, targetcols\
            = getDatasetInfo_dyn(df, datasetInfo)
    else:
        featuremapper, \
            targetmapper, \
            target_names, \
            isregression, \
            featurecols, \
            targetcols, \
            colmap = getDatasetInfo(dataset,
                                        multilabelbin=multilabelbin,
                                        n_splits=n_splits,
                                        df=df,
                                        datasetInfo=datasetInfo,
                                        n_samples=n_samples,
                                        n_features=n_features,
                                        colnames=colnames,
                                        classcols=classcols,
                                        targetcols=targetcols,
                                        featurecols=featurecols,
                                        targetcolindexes=targetcolindexes,
                                        dropcols=dropcols,
                                        colmap=colmap)

    # calculate the k-fold before converting the target column into e.g.
    # one-hot
    stratified_fold_generator = None
    if n_splits is not None:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        stratified_fold_generator = kfold.split(df.loc[:, featurecols].values,
                                                df.loc[:, targetcols].values)
    else:
        stratified_fold_generator = makeFakeYield(df.loc[:, featurecols].values,
                                                  df.loc[:, targetcols].values)

    # convert the data to proper format
    if "adapt" in datasetInfo:
        adapt_dyn = datasetInfo["adapt"]
        features, targets, colnames = adapt_dyn(df, datasetInfo)
        #colnames = featurecols.copy().extend(targetcols.copy())
    else:
        features, targets, colnames = adapt(df, featuremapper, targetmapper,
                                            featurecols, targetcols,
                                            datasetInfo, colmap)

    totaldata = np.append(features, targets, axis=1)
    featurecolsfrom = 0
    featurecolsto = features.shape[1]-1
    targetcolsfrom = features.shape[1]
    targetcolsto = features.shape[1]+targets.shape[1]
    return SKLearnDataset(totaldata, featurecolsfrom, featurecolsto,
                          targetcolsfrom, targetcolsto,
                          colnames, target_names, dataset,
                          isregression, targetcols), colmap, \
                          stratified_fold_generator
            #Bunch(data=features,target=targets,target_names=target_names,feature_names=feature_names)
