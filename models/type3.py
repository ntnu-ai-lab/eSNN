import keras
from keras import Input, Model
from keras.layers import Dense

from dataset.makeTrainingData import makeSmartNData
from models.eval import eval_normal_ann_l2, eval_chopra_ann
from models.utils import normalizeBatchSize, euclidean_distance, \
    eucl_dist_output_shape, makeAndCompileNormalModel
from utils.KerasCallbacks import callbackdict, CustomModelCheckPoint

# t3i1 trains G(x) on dataset used in eval with L2 to calculate distance between datapoints
from utils.keras_utils import contrastive_loss


def makeNormalArch(o_X, o_Y, X, Y, datasetname, regression=False, epochs=2000, val_split=0, shuffle=True, batch_size=32,
                  optimizer=None, onehot=True, multigpu=False, callbacks = None, networklayers=[13,13],
                  rootdir="rootdir",alpha=0.8, makeTrainingData=None):

    if isinstance(networklayers[0],list):
        # this means user has specified different layers
        # for g and c.. but we do not care as we only have layers for G(x)
        networklayers = networklayers[0]
    model,last_layer = makeAndCompileNormalModel(Y.shape[1], X.shape[1],
                                   optimizer=optimizer,regression=regression,
                                   onehot=onehot,multigpu=multigpu,
                                   activation_function="relu", networklayers=networklayers)



    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = X.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    filepath = rootdir + "gabelmodel"
    run_callbacks = list()
    ret_callbacks = dict()
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_normal_ann_l2,
                                                 datasetname,filepath,save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo
    history = model.fit(X, Y, validation_split=val_split,
                        shuffle=shuffle, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=run_callbacks)


    return model, history, ret_callbacks, model


def chopra(o_X, o_Y, X, Y, datasetname, regression=False, epochs=2000, val_split=0, shuffle=True,
                 batch_size=32, optimizer=None, onehot=True,
                 multigpu=False, callbacks=None, trainratio=0.2, networklayers=[13,13],
                 rootdir="rootdir",alpha=0.8, makeTrainingData=None):
    if makeTrainingData is None:
        makeTrainingData = makeSmartNData

    model, embeddingmodel = make_chopra_model(X, Y,
                                              networklayers, regression)

    features, targets, iX, iY = makeTrainingData(X, Y, regression, distance=False)

    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = features.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)

    model.compile(loss=contrastive_loss, optimizer=optimizer["constructor"](),
                      metrics=['accuracy'])
    training_data = [features[:,0:X.shape[1]],features[:,X.shape[1]:2*X.shape[1]]]
    filepath = rootdir + "dualshared-weights.best.hdf5"
    run_callbacks = list()
    ret_callbacks = dict()
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_chopra_ann,
                                                 datasetname, filepath, save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo
    filepath = rootdir+"saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
    run_callbacks.append(CustomModelCheckPoint(filepath="chopra", rootdir=rootdir))

    history = model.fit(training_data, targets,
              shuffle=True, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=run_callbacks)

    return model, history, ret_callbacks, embeddingmodel


def make_chopra_model(X, Y, networklayers, regression=False):

    g_layers = networklayers

    if isinstance(networklayers[0],list):
        # this means user has specified different layers
        # for g and c.. but chopra does not support this.. so c_layers
        # param will be ineffectual
        g_layers = networklayers[0]

    input1 = Input(shape=(X.shape[1],), dtype="float32")
    input2 = Input(shape=(X.shape[1],), dtype="float32")
    # make G(x)
    t1 = input1
    t2 = input2
    for networklayer in g_layers:
        dl1 = Dense(int(networklayer),
                    activation="relu", input_shape=t1.shape)
        # ,activity_regularizer=l2(0.01))
        t1 = dl1(t1)
        t2 = dl1(t2)
    # C(G(x),G(y) = L2(x,y)
    distance_layer = keras.layers.Lambda(euclidean_distance,
                                         output_shape=eucl_dist_output_shape)
    distance = distance_layer([t1, t2])
    model = Model(inputs=[input1, input2], outputs=[distance])
    embeddingmodel = Model(inputs=[input1],outputs=[t1])
    return model, embeddingmodel
