from dataset.dataset_to_sklearn import *
from models.eval import eval_dual_ann
from keras.layers import Dense, Input, Lambda
from keras.models import Model

from models.utils import normalizeBatchSize
from utils.keras_utils import keras_sqrt_diff
from dataset.makeTrainingData import makeSmartNData
from utils.KerasCallbacks import callbackdict,CustomModelCheckPoint


def esnn(o_X, o_Y, X, Y, datasetname, regression=False,
         epochs=2000, val_split=0, shuffle=True,
         batch_size=32, optimizer=None, onehot=True,
         multigpu=False, callbacks=None, trainratio=0.2,
         networklayers=[13, 13], rootdir="rootdir", alpha=0.8,
         makeTrainingData=None):

    if makeTrainingData is None:
        makeTrainingData = makeSmartNData

    model, embeddingmodel = make_eSNN_model(X, Y,
                                            networklayers, regression)
    #model.summary()

    #val_features, val_targets, val_Y1, val_Y2 = makeSmartNData(o_X, o_Y, regression)
    #

    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = X.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    if regression is not True:
        loss_dict = {#'dist_output': 'mean_squared_error',
                     'dist_output': 'binary_crossentropy',
                     'class1_output': 'categorical_crossentropy',
                     'class2_output': 'categorical_crossentropy'}
        lossweight_dict = {'dist_output': alpha,
                           'class1_output': (1.0-alpha)/2.0,
                           'class2_output': (1.0-alpha)/2.0}
    else:
        loss_dict = {'dist_output': 'mean_squared_error',
                     'reg_output1': 'mean_squared_error',
                     'reg_output2': 'mean_squared_error'}
        lossweight_dict = {'dist_output': alpha,
                           'reg_output1': (1.0-alpha)/2.0,
                           'reg_output2': (1.0-alpha)/2.0}

    model.compile(optimizer=optimizer["constructor"](),
                  loss=loss_dict,
                  metrics=['accuracy'],loss_weights=lossweight_dict)

    features, targets, Y1, Y2 = makeTrainingData(X, Y, regression, distance=True)
    training_data = [features[:,0:X.shape[1]],features[:,X.shape[1]:2*X.shape[1]]]

    target_data = [targets, Y1, Y2]


    run_callbacks = list()
    ret_callbacks = dict()
    filepath = rootdir + "esnn-weights.best.hdf5"
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_dual_ann,
                                                 datasetname, filepath, save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo
    filepath = rootdir+"saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
    run_callbacks.append(CustomModelCheckPoint(filepath="esnn", rootdir=rootdir))


    test = np.hstack((features,targets))
    batch_size = features.shape[0]
    history = model.fit(training_data, target_data,
                        shuffle=True, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=run_callbacks)#,
                        #validation_data=[val_training_data, val_target_data])

    return model, history, ret_callbacks, embeddingmodel


def make_eSNN_model(X, Y, networklayers, regression=False):
    g_layers = networklayers
    c_layers = networklayers

    if isinstance(networklayers[0],list):
        # this means user has specified different layers
        # for g and c..
        g_layers = networklayers[0]
        c_layers = networklayers[1]

    input1 = Input(shape=(X.shape[1],), dtype="float32")
    input2 = Input(shape=(X.shape[1],), dtype="float32")

    # make G(x)
    t1 = input1
    t2 = input2
    for networklayer in g_layers:
        dl1 = Dense(int(networklayer),
                    activation="relu",
                    input_shape=t1.shape)  # ,activity_regularizer=l2(0.01))
        t1 = dl1(t1)
        t2 = dl1(t2)

    dl1.name = "embeddinglayer"

    # subtracted = Subtract()([encoded_i1,encoded_i2])

    # TODO: We had 5 layers here (from subtract to output), maybe compare
    # different "top-layers" vs (the combination) "bottom-layers", in which
    # case we need more than one layers parameter

    # make C(x,y)
    o_t = Lambda(keras_sqrt_diff)([t1, t2])
    for networklayer in c_layers:
        o_t = Dense(int(networklayer), activation="relu")(o_t)

    # make class output from G(x) to get two more signal sources
    inner_output = None
    if regression is True:  # regression or 1 output classification
        inner_output1 = Dense(Y.shape[1], activation='linear',
                              kernel_initializer="random_uniform", name="reg_output1")
        inner_output2 = Dense(Y.shape[1], activation='linear',
                              kernel_initializer="random_uniform", name="reg_output2")
    else:  # onehot
        inner_output1 = Dense(Y.shape[1], activation='softmax', name="class1_output")
        inner_output2 = Dense(Y.shape[1], activation='softmax', name="class2_output")

    output = Dense(1, activation="sigmoid", name="dist_output")(o_t)

    output1 = inner_output1(t1)
    output2 = inner_output2(t2)

    model = Model(inputs=[input1, input2],
                  outputs=[output, output1, output2])

    embeddingmodel = Model(inputs=[input1],
                           outputs=[t1])
    return model, embeddingmodel
