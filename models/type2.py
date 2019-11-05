from keras import Input, Model
from keras.layers import Dense

from models.eval import eval_gabel_ann
from dataset.makeTrainingData import makeGabelTrainingData
from models.utils import normalizeBatchSize
from utils.KerasCallbacks import callbackdict


def makeGabelArch(o_X, o_Y, X, Y, datasetname, regression=False, epochs=2000, val_split=0, shuffle=True, batch_size=32,
                  optimizer=None, onehot=True, multigpu=False, callbacks = None, networklayers=[13,13],
                  rootdir="rootdir",alpha=0.8, makeTrainingData=None):
    #model = makeGabelClassifierModel(X.shape[1]*2, networklayers=[13,13]) #always 2 x 13 hidden layers
    if makeTrainingData is None:
        makeTrainingData = makeGabelTrainingData

    input1 = Input(shape=(X.shape[1]*2,), dtype="float32")

    hl1 = Dense(13, activation="sigmoid",kernel_initializer="random_uniform")(input1)
    hl2 = Dense(13, activation="sigmoid",kernel_initializer="random_uniform")(hl1)
    hl3 = Dense(13, activation="sigmoid",kernel_initializer="random_uniform")(hl2)
    output = Dense(1, activation="sigmoid",kernel_initializer="random_uniform")(hl3)
    model = Model(inputs=[input1],outputs=[output])
    model.compile(optimizer=optimizer["constructor"](),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    gabel_features, gabel_targets, Y1, Y2 = makeTrainingData(X, Y, regression, distance=False)
    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = gabel_features.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    filepath = rootdir + "gabelmodel"
    run_callbacks = list()
    ret_callbacks = dict()
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_gabel_ann,
                                                 datasetname,filepath,save_best_only=True,gabel=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo
    history = model.fit(gabel_features, gabel_targets, validation_split=val_split,
                        shuffle=shuffle, epochs=epochs, batch_size=gabel_features.shape[0],
                        verbose=0, callbacks=run_callbacks)
    return model, history, ret_callbacks, None # this type does not support embeddings as G(X) = I(X) = X