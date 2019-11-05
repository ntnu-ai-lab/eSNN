import pandas as pd
import numpy as np
import collections
import pickle
import os
import json
import urllib

import requests
import requests_cache
from sklearn import preprocessing

requests_cache.install_cache('bmm_uci_ml')
from io import StringIO
from pandas_datareader.compat import bytes_to_str
import xlrd
from dataset.binarizer import MyLabelBinarizer
__name__ = "dataset"
"""
from https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
"""

def writedatasetobjects(datasetsoboject):
    with open("datasets.pickle", 'wb') as f:
        json.dump(datasetsoboject.__dict__, f)


def readdatasetobjects(picklefilename):
    obj = Datasets()
    if not os.path.isfile(picklefilename):
        print("pickefile: %s is not here" % picklefilename)
        return False
    with open(os.path.join(picklefilename), 'rb') as picklefile:
        obj.__dict__ = pickle.load(picklefile)
    return obj

def glass_preprocess(df):
    colname = "type_of_glass"
    dffff = pd.DataFrame(data={})
    #df.loc[:,colname] = df.loc[:,colname].apply(pd.to_numeric)
    down_query_index = df.query('type_of_glass == "1" or type_of_glass == "2" or type_of_glass == "3" or type_of_glass == "4"').index
    up_query_index = df.query('type_of_glass == "5" or type_of_glass == "6" or type_of_glass == "7"').index


    df.iloc[down_query_index,-1] = "0"
    df.iloc[up_query_index,-1] = "1"
    return df


def mnist_getInfo(df, datasetInfo):
    values = df.values
    features = values[:,:-1]
    targets = values[:,-1:]
    target_names = []
    isregression = False
    targetcols = [features.shape[1]]
    featurecols = [i for i in range(0,features.shape[1])]

    return target_names, isregression, featurecols, targetcols


def mnist_adapt(df, datasetInfo):
    values = df.values
    features = values[:,:-1]
    # features.shape 70000,784
    targets = values[:,-1:]
    mlb = MyLabelBinarizer()
    targets = mlb.fit_transform(targets)
    # features.shape 70000,10
    #targetcols = [i for i in range(features.shape[1],features.shape[1]+targets.shape[1])]
    targetcols = [i for i in range(features.shape[1],features.shape[1]+targets.shape[1])]
    featurecols = [i for i in range(0,features.shape[1])]
    return features, targets, featurecols.extend(targetcols)

def flatten_mnist(x):
    dim = x.shape[1] * x.shape[2]
    output = np.zeros((x.shape[0],dim))
    for i in range(0, x.shape[0]):
        output[i,:] = x[i].flatten()
    return output

def mnist_makedf(url, datasetInfo):

    from keras.utils.data_utils import get_file
    path = get_file("mnist.npz",
                    origin=url)

    f = np.load(path)
    #datasetInfo["classwidth"] = f['y_train'].shape[1]
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    flattentrain = flatten_mnist(x_train)
    flattentest = flatten_mnist(x_test)
    all_train = np.concatenate((flattentrain, y_train.reshape((y_train.shape[0],1))), axis=1)
    all_test = np.concatenate((flattentest, y_test.reshape((y_test.shape[0],1))), axis=1)
    all_data = np.concatenate((all_train, all_test), axis=0)

    feature_values = all_data[:, :-1].astype("float32")
    feature_values /= 255
    target_values = all_data[:,-1]
    target_values = target_values.reshape((target_values.shape[0],1))
    all_normalized_data = np.concatenate((feature_values,target_values),axis=1)
    retdf = pd.DataFrame(data=all_normalized_data)

    return retdf

def ttt_preprocess(df):

    columns = ["top_left", "top_middle", "top_right",
               "middle_left", "middle_middle", "middle_right",
               "bottom_left"]
    for col in columns:
        le = preprocessing.LabelEncoder()
        le.fit()

    return df



def ttt_postprocess(nparr,colmap):
    """
    from https://gist.github.com/mmmayo13/b52cc0e48aa10e1c0eac54e9989d36de
    """
    #nparr = np.delete(nparr, [0,3,6,9,12,15,18,21,24], axis=1)

    return nparr,colmap

datamap = {
    "soybean-small": {
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
        "descriptionUrl": "https://archive.ics.uci.edu/ml/datasets/soybean+(small)",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "cols": [{"name": "date", "type": "nominal", "class": False},
                 {"name": "plant-stand", "type": "nominal", "class": False},
                 {"name": "precip", "type": "nominal", "class": False},
                 {"name": "temp", "type": "nominal", "class": False},
                 {"name": "hail", "type": "nominal", "class": False},
                 {"name": "crop-hist", "type": "nominal", "class": False},
                 {"name": "area-damaged", "type": "nominal", "class": False},
                 {"name": "severity", "type": "nominal", "class": False},
                 {"name": "seed-tmt", "type": "nominal", "class": False},
                 {"name": "germination", "type": "nominal", "class": False},
                 {"name": "plant-growth", "type": "nominal", "class": False},
                 {"name": "leaves", "type": "nominal", "class": False},
                 {"name": "leafspots-halo", "type": "nominal", "class": False},
                 {"name": "leafspots-marg", "type": "nominal", "class": False},
                 {"name": "leafspot-size", "type": "nominal", "class": False},
                 {"name": "leaf-shread", "type": "nominal", "class": False},
                 {"name": "leaf-malf", "type": "nominal", "class": False},
                 {"name": "leaf-mild", "type": "nominal", "class": False},
                 {"name": "stem", "type": "nominal", "class": False},
                 {"name": "lodging", "type": "nominal", "class": False},
                 {"name": "stem-cankers", "type": "nominal", "class": False},
                 {"name": "canker-lesion", "type": "nominal", "class": False},
                 {"name": "fruiting-bodies", "type": "nominal", "class": False},
                 {"name": "external decay", "type": "nominal", "class": False},
                 {"name": "mycelium", "type": "nominal", "class": False},
                 {"name": "int-discolor", "type": "nominal", "class": False},
                 {"name": "sclerotia", "type": "nominal", "class": False},
                 {"name": "fruit-pods", "type": "nominal", "class": False},
                 {"name": "fruit spots", "type": "nominal", "class": False},
                 {"name": "seed", "type": "nominal", "class": False},
                 {"name": "mold-growth", "type": "nominal", "class": False},
                 {"name": "seed-discolor", "type": "nominal", "class": False},
                 {"name": "seed-size", "type": "nominal", "class": False},
                 {"name": "shriveling", "type": "nominal", "class": False},
                 {"name": "roots", "type": "nominal", "class": False},
                 {"name": "class", "type": "nominal", "class": True}
        ]

    },
    "bal": {
        "dataUrl": "http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "cols": [{"name": "class", "type": "nominal", "class": True},
                 {"name": "left_weight", "type": "nominal", "class": False},
                 {"name": "left_distance", "type": "nominal", "class": False},
                 {"name": "right_weight", "type": "nominal", "class": False},
                 {"name": "right_distance", "type": "nominal", "class": False}
        ]
    },
    "car": {
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "cols": [{"name": "bying", "type": "nominal", "class": False},
                 {"name": "maint", "type": "nominal", "class": False},
                 {"name": "doors", "type": "nominal", "class": False},
                 {"name": "persons", "type": "nominal", "class": False},
                 {"name": "lug_boot", "type": "nominal", "class": False},
                 {"name": "safety", "type": "nominal", "class": False},
                 {"name": "class", "type": "nominal", "class": True}
        ]
    },
    "cmc": {
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "cols": [{"name": "wife_age", "type": np.int32, "class": False},
                 {"name": "wife_education", "type": "nominal", "class": False},
                 {"name": "husbands_education", "type": "nominal", "class": False},
                 {"name": "number_of_children_born", "type": "nominal", "class": False},
                 {"name": "wife_religion", "type": "nominal", "class": False},
                 {"name": "wife_working", "type": "nominal", "class": False},
                 {"name": "husbands_occupation", "type": "nominal", "class": False},
                 {"name": "standard_of_living", "type": "nominal", "class": False},
                 {"name": "media_exposure", "type": "nominal", "class": False},
                 {"name": "contraceptive", "type": "nominal", "class": True}
        ]
    },
    "eco": {
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
        "headers": None,
        "sep": "\s+",
        "num_classes": 1,
        "cols": [{"name": "skip", "type": "skip", "class": False},
                 {"name": "mcg", "type": np.float32, "class": False},
                 {"name": "gvh", "type": np.float32, "class": False},
                 {"name": "lip", "type": np.float32, "class": False},
                 {"name": "chg", "type": np.float32, "class": False},
                 {"name": "aac", "type": np.float32, "class": False},
                 {"name": "a1m1", "type": np.float32, "class": False},
                 {"name": "a1m2", "type": np.float32, "class": False},
                 {"name": "localization_site", "type": "nominal", "class": True}
        ]
    },
    "glass": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "sep": ",",
        "headers": None,
        "num_classes": 1,
        "pre_process": glass_preprocess,
        "cols": [{"name": "skip", "type": "skip", "class": False},
                 {"name": "RI", "type": np.float32, "class": False},
                 {"name": "Na", "type": np.float32, "class": False},
                 {"name": "Mg", "type": np.float32, "class": False},
                 {"name": "Al", "type": np.float32, "class": False},
                 {"name": "Si", "type": np.float32, "class": False},
                 {"name": "K", "type": np.float32, "class": False},
                 {"name": "Ca", "type": np.float32, "class": False},
                 {"name": "Ba", "type": np.float32, "class": False},
                 {"name": "Fe", "type": np.float32, "class": False},
                 {"name": "type_of_glass", "type": "nominal", "class": True}
        ]
    },
    "hay": {
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.data",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "cols": [{"name": "name", "type": "skip", "class": False},
                 {"name": "hobby", "type": "nominal", "class": False},
                 {"name": "age", "type": "nominal", "class": False},
                 {"name": "educational_level", "type": "nominal", "class": False},
                 {"name": "marital_status", "type": "nominal", "class": False},
                 {"name": "class", "type": "nominal", "class": True}
        ]
    },
    "heart": {
        "dataUrl": "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
        "headers": None,
        "sep": "\s+",
        "num_classes": 1,
        "cols": [{"name": "age", "type": np.float32, "class": False},
                 {"name": "sex", "type": "nominal", "class": False},
                 {"name": "chest_pain_type", "type": "nominal", "class": False},
                 {"name": "resting_blood_pressure", "type": np.float32, "class": False},
                 {"name": "serum_cholestroal", "type": np.float32, "class": False},
                 {"name": "fasting_blood_sugar", "type": "nominal", "class": False},
                 {"name": "resting_ecg", "type": "nominal", "class": False},
                 {"name": "max_hr", "type": np.float32, "class": False},
                 {"name": "exercise_agina", "type": "nominal", "class": False},
                 {"name": "oldpeak", "type": np.float32, "class": False},
                 {"name": "slope", "type": "nominal", "class": False},
                 {"name": "numb_blood_vessels", "type": np.float32, "class": False},
                 {"name": "thal", "type": "nominal", "class": False},
                 {"name": "heartdefect", "type": "nominal", "class": True}
        ]
    },
    "iris": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "backupUrl":"./dataset/iris.csv",
        "sep": ",",
        "headers": None,
        "num_classes": 1,
        "cols": [
            {"name": "sepal_length", "type": np.float32, "class": False},
            {"name": "sepal_width", "type": np.float32, "class": False},
            {"name": "petal_length", "type": np.float32, "class": False},
            {"name": "petal_width", "type": np.float32, "class": False},
            {"name": "type_of_iris", "type": "nominal", "class": True}
        ]
    },
    "mam": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
        "sep": ",",
        "headers": None,
        "na_values": "?",
        "num_classes": 1,
        "dropcols": ["BI-RADS"],
        "dropna": True,
        "cols": [
            {"name": "BI-RADS", "type": "nominal","dtype":np.int32, "class": False},
            {"name": "Age", "type": np.float32, "class": False},
            {"name": "Shape", "type": "nominal", "class": False},
            {"name": "Margin", "type": "nominal", "class": False},
            {"name": "Density", "type": "nominal", "class": False},
            {"name": "Severity", "type": "nominal", "class": True}
        ]
    },
    "mon": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train",
        "sep": "\s+",
        "headers": None,
        "num_classes": 1,
        "cols": [
            {"name": "class", "type": "nominal", "class": True},
            {"name": "a1", "type": np.int32, "class": False},
            {"name": "a2", "type": np.int32, "class": False},
            {"name": "a3", "type": np.int32, "class": False},
            {"name": "a4", "type": np.int32, "class": False},
            {"name": "a5", "type": np.int32, "class": False},
            {"name": "a6", "type": np.int32, "class": False},
            {"name": "skip", "type": "skip", "class": False}
        ]
    },
    "pim": {
        "dataUrl":"https://raw.githubusercontent.com/LamaHamadeh/Pima-Indians-Diabetes-DataSet-UCI/master/pima_indians_diabetes.txt",
        "descriptionUrl": "https://github.com/LamaHamadeh/Pima-Indians-Diabetes-DataSet-UCI",
        "headers": 0,
        "sep": ",",
        "num_classes": 2,
        "cols": [
            {"name": "No_pregnant", "type": np.float32, "class": False},
            {"name": "Plasma_glucose", "type": np.float32, "class": False},
            {"name": "Blood_pres", "type": np.float32, "class": False},
            {"name": "Skin_thick", "type": np.float32, "class": False},
            {"name": "Serum_insu", "type": np.float32, "class": False},
            {"name": "BMI", "type": np.int32, "class": False},
            {"name": "Diabetes_func", "type": np.float32, "class": False},
            {"name": "Age", "type": np.float32, "class": False},
            {"name": "Class", "type": "nominal", "class": True}
        ]
    },
    "energy": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "descriptionUrl": "https://archive.ics.uci.edu/ml/datasets/Energy+efficiency#",
        "headers": 0,
        "sheet_name": [0],
        "num_classes": 2,
        "cols": [
            {"name": "X1", "type": np.float32, "class": False},
            {"name": "X2", "type": np.float32, "class": False},
            {"name": "X3", "type": np.float32, "class": False},
            {"name": "X4", "type": np.float32, "class": False},
            {"name": "X5", "type": np.float32, "class": False},
            {"name": "X6", "type": np.int32, "class": False},
            {"name": "X7", "type": np.float32, "class": False},
            {"name": "X8", "type": np.float32, "class": False},
            {"name": "Y1", "type": np.float32, "class": True},
            {"name": "Y2", "type": np.float32, "class": True}
        ]
    },
    "ttt":{
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "post_process": ttt_postprocess,
        "cols": [{"name": "top_left", "type": "nominal", "class": False},
                 {"name": "top_middle", "type": "nominal", "class": False},
                 {"name": "top_right", "type": "nominal", "class": False},
                 {"name": "middle_left", "type": "nominal", "class": False},
                 {"name": "middle_middle", "type": "nominal", "class": False},
                 {"name": "middle_right", "type": "nominal", "class": False},
                 {"name": "bottom_left", "type": "nominal", "class": False},
                 {"name": "bottom_middle", "type": "nominal", "class": False},
                 {"name": "bottom_right", "type": "nominal", "class": False},
                 {"name": "class", "type": "nominal", "class": True}
        ]
    },
    "use": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/00257/Data_User_Modeling_Dataset_Hamdi%20Tolga%20KAHRAMAN.xls",
        "headers": 0,
        "sheet_name": [1,2],#["Training_Data","Test_Data"],
        "usecols": "A:F",
        "num_classes": 1,
        "cols": [
            {"name": "STG", "type": np.int32, "class": False},
            {"name": "SCG", "type": np.int32, "class": False},
            {"name": "STR", "type": np.int32, "class": False},
            {"name": "LPR", "type": np.int32, "class": False},
            {"name": "PEG", "type": np.int32, "class": False},
            {"name": "UNS", "type": "nominal", "class": True}
        ]
    },
    "who": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv",
        "sep": ",",
        "headers": 0,
        "num_classes": 2,
        "cols": [
            {"name": "Channel", "type": "nominal", "class": True},
            {"name": "Region", "type": "nominal", "class": True},
            {"name": "Fresh", "type": np.int32, "class": False},
            {"name": "Milk", "type": np.int32, "class": False},
            {"name": "Grocery", "type": np.int32, "class": False},
            {"name": "Frozen", "type": np.int32, "class": False},
            {"name": "Detergents_Paper", "type": np.int32, "class": False},
            {"name": "Delicatessen", "type": np.int32, "class": False}
        ]#It seems the target variable is "region" or "channel"
    },
    "bupa": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data",
        "descriptionUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.names",
        "sep": ",",
        "headers": None,
        "num_classes": 1,
        "cols": [
            {"name": "mcv", "type": np.int32, "class": False},
            {"name": "alkphos", "type": np.int32, "class": False},
            {"name": "sgpt", "type": np.int32, "class": False},
            {"name": "sgot", "type": np.int32, "class": False},
            {"name": "gammagt", "type": np.int32, "class": False},
            {"name": "drinks", "type": np.float32, "class": False},
            {"name": "selector", "type": "nominal", "dtype": "str", "class": True}
        ]
    },
    "housing": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "descriptionUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names",
        "sep": "\s+",
        "headers": None,
        "num_classes": 1,
        "cols": [
            {"name": "CRIM", "type": np.float32, "class": False},
            {"name": "ZN", "type": np.float32, "class": False},
            {"name": "INDUS", "type": np.float32, "class": False},
            # this is nominal but only contains 0 or 1
            {"name": "CHAS", "type": np.int32, "class": False},
            {"name": "NOX", "type": np.float32, "class": False},
            {"name": "RM", "type": np.float32, "class": False},
            {"name": "AGE", "type": np.float32, "class": False},
            {"name": "DIS", "type": np.float32, "class": False},
            {"name": "RAD", "type": np.int32, "class": False},
            {"name": "TAX", "type": np.float32, "class": False},
            {"name": "PTRATIO", "type": np.float32, "class": False},
            {"name": "B", "type": np.float32, "class": False},
            {"name": "LSTAT", "type": np.float32, "class": False},
            {"name": "MEDV", "type": np.float32, "class": True}
        ]
    },
    "machine": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
        "descriptionUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.names",
        "sep": ",",
        "headers": None,
        "num_classes": 1,
        "cols": [            #skip 3 parameters given the description above
            {"name": "vendor_name", "type": "skip", "class": False},
            {"name": "model_name", "type": "skip", "class": False},
            {"name": "MYCT", "type": np.float32, "class": False},
            {"name": "MMIN", "type": np.float32, "class": False},
            {"name": "MMAX", "type": np.float32, "class": False},
            {"name": "CACH", "type": np.float32, "class": False},
            {"name": "CHMIN", "type": np.float32, "class": False},
            {"name": "CHMAX", "type": np.float32, "class": False},
            {"name": "PRP", "type": np.float32, "class": True},
            {"name": "ERP", "type": "skip", "class": True}
            ]
    },
    "servo": {
        "dataUrl": "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data",
        "headers": None,
        "sep": ",",
        "num_classes": 1,
        "cols": [{"name": "motor", "type": "nominal", "class": False},
                 {"name": "screw", "type": "nominal", "class": False},
                 {"name": "pgain", "type": "nominal", "class": False},
                 {"name": "vgain", "type": "nominal", "class": False},
                 {"name": "class", "type": np.float32, "class": True}
        ]
    },
    "yacht": {
        "dataUrl":"https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        "sep": "\s+",
        "headers": None,
        "num_classes": 1,
        "cols": [
            {"name": "lon", "type": np.float32, "class": False},
            {"name": "primsatic_coeff", "type": np.float32, "class": False},
            {"name": "length-displacement", "type": np.float32, "class": False},
            {"name": "beam-draught_ratio", "type": np.float32, "class": False},
            {"name": "length-beam_ratio", "type": np.float32, "class": False},
            {"name": "froude_number", "type": np.float32, "class": False},
            {"name": "residuary_resistance", "type": np.float32, "class": True}
        ]
    },
    # this dataset is not a csv and needs a lot of code to adapt to the same
    # format as the others.
    "mnist": {
        "dataUrl": "https://s3.amazonaws.com/img-datasets/mnist.npz",
        "headers": None,
        "makedf": mnist_makedf,
        "num_classes": 1,
        "adapt": mnist_adapt,
        "getInfo": mnist_getInfo
    }
}


# inspiration from https://github.com/pydata/pandas-datareader/blob/master/pandas_datareader/base.py
# and https://github.com/davidastephens/pandas-finance/blob/master/pandas_finance/api.py

def safeGetDF(url):
    httpsession = requests_cache.CachedSession(cache_name='bmm-uciml-cache', backend='sqlite')
    response = httpsession.get(url)
    if response.status_code != requests.codes.ok:
        return None
    text = response.content
    out = StringIO()
    if len(text) == 0:
        #service = self.__class__.__name__
        raise IOError("{} request returned no data; check URL for invalid ")
    if ".xls" in url:
        httpsession.close()
        return xlrd.open_workbook(file_contents=text)
        #out.write(text)
        #out.write(bytes_to_str(text,encoding="ISO-8859-1"))
    elif response.headers["content-type"] != "text/html":
        out.write(bytes_to_str(text,encoding="ISO-8859-1"))
    else:
        out.write(text)
    out.seek(0)
    httpsession.close()
    #decoded_data = out.decode("utf-8")
    return out

def convert(inp):
    if inp is "nominal" or inp is "ordinal" or inp:
        return "str"
    elif inp is "binary":
        return np.int32
    else:
        return inp

def makePosList(inp):
    counter = 0
    ret = []
    for el in inp:
        if el["type"] is "skip":
            counter += 1
        else:
            ret.append(counter)
            counter += 1
    return ret

class Dataset():
    def __init__(self, key):
        self.loadDataset(key)
        self.name = key

    def readFromUrl(self, url, sep=" "):
        self.df = pd.read_csv(url, sep, header=None)


    def loadDataset(self, key):
        self.datasetInfo = datamap[key]
        url = self.datasetInfo["dataUrl"]
        #print("loading dataset {}".format(key))
        if "makedf" in self.datasetInfo:
            self.df = self.datasetInfo["makedf"](url,self.datasetInfo)
            return
        usecols = makePosList(self.datasetInfo["cols"])
        colnames = [c["name"] for c in self.datasetInfo["cols"] if c["type"] is not "skip"]
        dtypes = [c["type"] for c in self.datasetInfo["cols"] if c["type"] is not "skip"]
        dtypes = map(convert,dtypes)
        dtypedict = dict(zip(colnames,dtypes))
        #print(dtypedict)

        try:
            #url = self.datasetInfo["backupUrl"]
            r = requests.get(url)
            #r.raise_for_status()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(f"{e}")
        except requests.exceptions.HTTPError as httperror:
            print(f"got error for dataset {key}")
            url = self.datasetInfo["backupUrl"]


        if "xls" in url:
            sheet_names = self.datasetInfo["sheet_name"]
            excelusecols = ",".join(colnames)
            if "usecols" in self.datasetInfo:
                possibledictofdf = pd.read_excel(safeGetDF(url),
                                                 header=self.datasetInfo["headers"],
                                                 sheet_name=sheet_names,
                                                 names=colnames,
                                                 dtype=dtypedict,
                                                 usecols=self.datasetInfo["usecols"],
                                                 engine="xlrd")
            else:
                possibledictofdf = pd.read_excel(safeGetDF(url),
                                                 header=self.datasetInfo["headers"],
                                                 sheet_name=sheet_names,
                                                 names=colnames,
                                                 dtype=dtypedict, engine="xlrd")
            if isinstance(sheet_names, collections.Sequence) and not isinstance(sheet_names, str):
                firstdf = possibledictofdf[sheet_names[0]]
                for sheet in sheet_names:
                    if sheet is not sheet_names[0]:
                        firstdf.append(possibledictofdf[sheet])
                self.df = firstdf[colnames]
            else:
                self.df = possibledictofdf[colnames]
        else: #this is a csv file
            na_values = None
            if "na_values" in self.datasetInfo:
                na_values = self.datasetInfo["na_values"]
            self.df = pd.read_csv(safeGetDF(url), header=self.datasetInfo["headers"],
                                  names=colnames, dtype=dtypedict,
                                  usecols=usecols,
                                  sep=self.datasetInfo["sep"],
                                  na_values=na_values)
        if "pre_process" in self.datasetInfo:
            self.df = self.datasetInfo["pre_process"](self.df)
            #print(self.df)

    def getNumberOfRows(self):
        return self.df.shape[0]

    def getNumberOfAttributes(self):
        return self.df.shape[1]-1

    def getTypes(self):
        return self.datasetInfo["cols"]

    def getMaxForCol(self, col):
        return self.df[col].max()

    def getMinForCol(self, col):
        return self.df[col].min()


class Datasets():
    def __init__(self, filename=None):
        self.datasets = {}
        if filename is not None:
            if not os.path.isfile(filename):
                print(f"pickefile: {filename} is not here")
                return
            with open(os.path.join(filename), 'rb') as picklefile:
                self.__dict__ = pickle.load(picklefile)

    def getDataset(self, key):
        if key not in self.datasets:
            ds = Dataset(key)
            self.datasets[key] = ds
            writedatasetobjects(self)
            return ds
        return self.datasets[key]
