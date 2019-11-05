import numpy as np
import math
from sklearn.preprocessing.label import _inverse_binarize_multiclass
import random

def eucdistanceInstance(i1, i2, colname):
    v1 = i1.instance_parameters[colname]
    v2 = i2.instance_parameters[colname]
    return eucdistance(v1, v2)

def eucdistance(v1, v2):
    return math.sqrt(math.pow(v1-v2, 2))

def makeGabelTrainingData(features, targets, isregression, distance = False):
    #features = dsl.getFeatures()

    #targets = dsl.getTargets()
    #make the two 
    Y1 = np.zeros((features.shape[0] ** 2, targets.shape[1]))
    Y2 = np.zeros((features.shape[0] ** 2, targets.shape[1]))
    combineddata = np.zeros((features.shape[0] ** 2, 2 * features.shape[1]
                             + ((targets.shape[1] * 2) + 1)))
    targetout = np.zeros([targets.shape[0] ** 2, 1])
    if isregression:
        for i in range(features.shape[0]):
            tile = np.tile(features[i], (features.shape[0], 1))
            combineddata[i*features.shape[0]:(i*features.shape[0])
                         + features.shape[0]
                         , 0:features.shape[1]] = tile
            combineddata[i * features.shape[0]:(i * features.shape[0])
                         + features.shape[0]
                         , features.shape[1]:(features.shape[1]*2)] = features
            for i2 in range(targets.shape[0]):
                targetout[(i*features.shape[0])+i2, 0] = 1.0
                - eucdistance(targets[i], targets[i2])
    else:
        for i in range(features.shape[0]):
            tile = np.tile(features[i], (features.shape[0], 1))
            combineddata[i * features.shape[0]:(i * features.shape[0]) + features.shape[0]
            , 0:features.shape[1]] = tile
            combineddata[i * features.shape[0]:(i * features.shape[0]) + features.shape[0]
            , features.shape[1]:(features.shape[1] * 2)] = features

            combineddata[(i * features.shape[0]):((i + 1) * features.shape[0]),
            (features.shape[1] * 2):(features.shape[1] * 2) + targets.shape[1]] = targets

            combineddata[(i * features.shape[0]):((i + 1) * features.shape[0]),
            (features.shape[1] * 2) + targets.shape[1]:(features.shape[1] * 2) + (targets.shape[1]*2)] = np.tile(targets[i],
                                                                               (features.shape[0], 1))
            for i2 in range(targets.shape[0]):
                counter = 0
                Y1[(i * features.shape[0])+i2, 0:targets.shape[1]] = targets[i]
                Y2[(i * features.shape[0])+i2, 0:targets.shape[1]] = targets[i2]
                #for i3 in range(targets.shape[1]):
                #    if targets[i][i3] == targets[i2][i3]:
                #        counter += 1
                compared = easycompare(targets[i],targets[i2])
                targetout[(i * features.shape[0])+i2, 0] = compared

            #print("hmm")

    combineddata[:,
    (features.shape[1] * 2) + (targets.shape[1] * 2):(features.shape[1] * 2) + (targets.shape[1]*2) + 1] = targetout
    
    return combineddata[:,0:features.shape[1]*2],\
           combineddata[:,(features.shape[1] * 2) + (targets.shape[1] * 2):
                          (features.shape[1] * 2) + (targets.shape[1]*2) + 1],Y1,Y2



def makeDualSharedArchData(features, targets, isregression, distance = False):
    #features = dsl.getFeatures()
    #targets = dsl.getTargets()
    Y1 = np.zeros((features.shape[0] ** 2, targets.shape[1]))
    Y2 = np.zeros((features.shape[0]**2,targets.shape[1]))
    combineddata = np.zeros((features.shape[0]**2,2*features.shape[1]+((targets.shape[1]*2)+1)))
    targetout = np.zeros([targets.shape[0]**2,1])
    if isregression:
        for i in range(features.shape[0]):
            tile = np.tile(features[i], (features.shape[0], 1))
            combineddata[i*features.shape[0]:(i*features.shape[0])+features.shape[0]
            ,0:features.shape[1]] = tile
            combineddata[i * features.shape[0]:(i * features.shape[0]) + features.shape[0]
            , features.shape[1]:(features.shape[1]*2)] = features
            for i2 in range(targets.shape[0]):
                targetout[(i*features.shape[0])+i2, 0] = 1.0-eucdistance(targets[i], targets[i2])
    else:
        for i in range(features.shape[0]):
            tile = np.tile(features[i], (features.shape[0], 1))
            combineddata[i * features.shape[0]:(i * features.shape[0]) + features.shape[0]
            , 0:features.shape[1]] = tile
            combineddata[i * features.shape[0]:(i * features.shape[0]) + features.shape[0]
            , features.shape[1]:(features.shape[1] * 2)] = features

            combineddata[(i * features.shape[0]):((i + 1) * features.shape[0]),
            (features.shape[1] * 2):(features.shape[1] * 2) + targets.shape[1]] = targets

            combineddata[(i * features.shape[0]):((i + 1) * features.shape[0]),
            (features.shape[1] * 2) + targets.shape[1]:(features.shape[1] * 2) + (targets.shape[1]*2)] = np.tile(targets[i],
                                                                               (features.shape[0], 1))

            for i2 in range(targets.shape[0]):
                counter = 0
                Y1[(i * features.shape[0])+i2,0:targets.shape[1]] = targets[i]
                Y2[(i * features.shape[0])+i2,0:targets.shape[1]] = targets[i2]
                #for i3 in range(targets.shape[1]):
                #    if targets[i][i3] != targets[i2][i3]:
                #        counter += 1
                compared = easycompare(targets[i],targets[i2])
                # 1.0-(counter/float(targets.shape[1]))
                targetout[(i * features.shape[0])+i2, 0] = 1.0-compared

    combineddata[:,
    (features.shape[1] * 2) + (targets.shape[1] * 2):(features.shape[1] * 2) + (targets.shape[1]*2) + 1] = targetout

    return combineddata[:,0:features.shape[1]*2],\
           combineddata[:,(features.shape[1] * 2) + (targets.shape[1] * 2):
                          (features.shape[1] * 2) + (targets.shape[1]*2) + 1], Y1, Y2

def comparetargets(target1, target2):
    counter = 0
    for i in range(target1.shape[0]):
        if target1[i] == target2[i]:
            counter += 1
    return counter / float(target1.shape[0])

def easycompare(target1, target2):
    return all(np.equal(target1,target2))

def makeNData(features, targets, isregression, distance = True):
    combineddata = np.zeros((features.shape[0],(2*features.shape[1])))
    #targetout = np.zeros([targets.shape[0],1])
    combineddata[:, 0:features.shape[1]] = features
    copieddata = np.concatenate((features.copy(), targets.copy()),axis=1)
    scrambled_copy = copieddata.copy()
    np.random.shuffle(scrambled_copy)
    #scrambled_copy = scrambled_copy[::-1]
    scrambled_features = scrambled_copy[:, :-targets.shape[1]]
    combineddata[:, features.shape[1]:features.shape[1]*2] = scrambled_features
    sim_target = np.zeros((targets.shape[0],1))
    for i in range(0, len(combineddata)):
        compared = easycompare(targets[i], scrambled_copy[i, features.shape[1]:])
        if distance:
            sim_target[i] = 1-compared
        else:
            sim_target[i] = compared

    return combineddata, sim_target, targets.copy(), \
        scrambled_copy[:, -targets.shape[1]:]


def getSameClass(data, thisclass, indices, num_classes, class_dict):
    dn = thisclass
    secondind = class_dict[dn]["counter"]
    class_max = class_dict[dn]["max"]
    class_dict[dn]["counter"] = (secondind+1) % class_max
    return data[indices[dn][secondind], :], indices[dn][secondind]


# pick another class than thisclass at random, then iterates over all examples
# of that class, not reusing example until it every other example also is used.
# no need to shuffle or jump randomly in the indexes of that class, as this
# data is already shuffled in stratified kfold
def getOtherClass(data, thisclass, indices, num_classes, class_dict):
    inc = random.randrange(1, num_classes)
    dn = (thisclass + inc) % num_classes
    secondind = class_dict[dn]["counter"]
    class_max = class_dict[dn]["max"]
    class_dict[dn]["counter"] = (secondind+1) % class_max
    return data[indices[dn][secondind], :], indices[dn][secondind]


def makeSmartNData(features, targets, isregression, distance = False):
    """
    Some comment..
    :param features:
    :param targets:
    :param isregression:
    :return:
    """
    num_classes = targets.shape[1]
    targetscopy = targets.copy()
    nonbinarized = _inverse_binarize_multiclass(targetscopy,
                                                [i for i in range(num_classes)])
    digit_indices = [np.where(nonbinarized == i)[0] for i in range(num_classes)]
    classcounter = {key:
                    {"counter": 1, "max": len(digit_indices[key])}
                    for key in range(0,num_classes)}
    combineddata = np.zeros((features.shape[0],(2*features.shape[1])))
    targets2 = np.zeros((targets.shape[0],targets.shape[1]))
    combineddata[:, 0:features.shape[1]] = features

    sim_target = np.zeros((targets.shape[0], 1))
    positive = True
    targetind = 0
    for i in range(0, targets.shape[0]):
        if positive:
            # this time pick other part of pair from same class (positive pair)
            newdata, targetind = getSameClass(features, nonbinarized[i],
                                              digit_indices, num_classes,
                                              classcounter)
            combineddata[i, features.shape[1]:features.shape[1]*2] = newdata
            if distance:
                sim_target[i] = 0
            else:
                sim_target[i] = 1
            positive = False
        else:
            # pick the other part of the pair from another
            # class (negative pair)
            newdata, targetind = getOtherClass(features, nonbinarized[i],
                                               digit_indices, num_classes,
                                               classcounter)
            combineddata[i, features.shape[1]:features.shape[1]*2] = newdata
            if distance:
                sim_target[i] = 1
            else:
                sim_target[i] = 0
            positive = True
        targets2[i] = targets[targetind]


    return combineddata, sim_target, targets.copy(), \
        targets2


def makeSemiBig(features, targets, isregressive, distance = False):
    sizeofarr = int((features.shape[0]**2)/2) + int((features.shape[0]/2)+0.5)
    combineddata = np.zeros((sizeofarr, (2*features.shape[1])))
    targets1 = np.zeros((sizeofarr, targets.shape[1]))
    targets2 = np.zeros((sizeofarr, targets.shape[1]))
    counter = 0
    sim_target = np.zeros((sizeofarr, 1))
    ibase = 0
    for i in range(0, features.shape[0]):
        comparedto = features.shape[0] - counter
        combineddata[ibase:ibase+comparedto, :features.shape[1]] = \
            np.tile(features[i], (comparedto, 1))
        targets1[ibase:ibase + comparedto,:] = \
            np.tile(targets[i], (comparedto, 1))
        for i2 in range(counter, features.shape[0]):
            combineddata[ibase+(i2-counter),
                         features.shape[1]:2*features.shape[1]] = features[i2]
            targets2[ibase + (i2 - counter)] = targets[i2]
            if all(np.equal(targets[i], targets[i2])):
                sim_target[ibase+(i2-counter)] = 0
            else:
                sim_target[ibase+(i2-counter)] = 1
        counter += 1
        ibase += comparedto
    return combineddata, sim_target, targets1, \
        targets2
