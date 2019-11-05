from keras import Model, Input
from keras.layers import Subtract, Dense, Lambda, Concatenate
from keras.models import clone_model

from models.eval import eval_dual_ann
from dataset.makeTrainingData import makeGabelTrainingData,\
    makeDualSharedArchData
from models.utils import normalizeBatchSize,  makeAndCompileNormalModel,\
    makeNormalModelLayers
from utils.KerasCallbacks import callbackdict, CustomModelCheckPoint
from utils.keras_utils import keras_sqrt_diff

#t4i4
def makeDualArch(o_x, o_Y, X, Y, datasetname, regression=False, epochs=2000, val_split=0, shuffle=True,
                 batch_size=32, optimizer=None, onehot=True,
                 multigpu=False, callbacks=None, trainratio=0.2, networklayers=[13,13],
                 rootdir="rootdir",alpha=0.8, makeTrainingData=None):
    if isinstance(networklayers[0],list):
        # this means user has specified different layers
        # for g and c.. but we do not care as we only have layers for G(x)
        networklayers = networklayers[0]

    model, output = makeAndCompileNormalModel(Y.shape[1], X.shape[1], optimizer=optimizer,
                                    regression=regression, onehot=onehot,
                                    multigpu=multigpu, activation_function="relu",
                                    networklayers=networklayers)

    if makeTrainingData is None:
        makeTrainingData = makeGabelTrainingData

    history = model.fit(X, Y, validation_split=val_split,
                        shuffle=shuffle, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=callbacks)

    layers = model.layers

    model_clone = clone_model(model)
    model_clone.set_weights(model.get_weights())

    clone_layers = model_clone.layers

    second_last_layer_orig = layers[len(layers)-2]
    second_last_layer_clone = clone_layers[len(layers)-2]

    subtracted = Subtract()([second_last_layer_orig.output,second_last_layer_clone.output])

    dense_1 = Dense(10, activation="relu")(subtracted)
    dense_2 = Dense(10, activation="relu")(dense_1)

    output = Dense(1, activation="linear")(dense_2)

    for layer in layers:
        layer.trainable = False

    for layer in clone_layers:
        layer.trainable = False
        layer.name = layer.name+"_clone"

    model = Model(inputs=[model.input,model_clone.input],outputs=[output])
    embeddingmodel = Model(inputs=[model.input],
                           outputs=[second_last_layer_orig])
    features, targets = makeTrainingData(X, Y, regression)

    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = features.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    run_callbacks = list()
    ret_callbacks = dict()
    filepath = rootdir + "t4i4-weights.best.hdf5"
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_dual_ann,
                                                 datasetname, filepath, save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo
    filepath = rootdir+"saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
    run_callbacks.append(CustomModelCheckPoint(filepath="t4i4", rootdir=rootdir))

    model.compile(loss='binary_cross_entropy', optimizer=optimizer["constructor"](),
                      metrics=['accuracy'])
    training_data = [features[:,0:X.shape[1]],features[:,X.shape[1]:2*X.shape[1]]]
    history = model.fit(training_data, targets,
              shuffle=True, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=run_callbacks)

    return model, history, ret_callbacks, embeddingmodel


def makeEndToEndDualArch(o_X, o_Y, X, Y, datasetname, regression=False, epochs=2000, val_split=0, shuffle=True,
                         batch_size=32, optimizer=None, onehot=True,
                         multigpu=False, callbacks=None, trainratio=0.2, networklayers=[13, 13],
                         rootdir="rootdir", alpha=0.8, makeTrainingData=None):

    input1,output1,layers1 = makeNormalModelLayers(n=Y.shape[1], inp=X.shape[1], networklayers=networklayers,
                          regression=regression, activation_function="relu")
    input2,output2,layers2 = makeNormalModelLayers(n=Y.shape[1], inp=X.shape[1], networklayers=networklayers,
                          regression=regression, activation_function="relu")

    if makeTrainingData is None:
        makeTrainingData = makeDualSharedArchData

    #history = model.fit(X, Y, validation_split=val_split,
    #                    shuffle=shuffle, epochs=epochs, batch_size=batch_size,
    #                    verbose=0, callbacks=callbacks)



    #clone_layers = model_clone.layers

    #second_last_layer_orig = layers[ln(layers)-2]
    #second_last_layer_clone = clone_layers[len(layers)-2]

    subtracted = Subtract()([layers1[len(layers1)-2],layers2[len(layers2)-2]])

    dense_1 = Dense(10, activation="relu")(subtracted)
    dense_2 = Dense(10, activation="relu")(dense_1)

    output = Dense(1, activation="relu")(dense_2)

    #for layer in layers2:
    #    layer.name = layer.name+"_clone"

    model = Model(inputs=[input1,input2],outputs=[output,output1,output2])


    # comb = np.zeros((features.shape[0],features.shape[1]+targets.shape[1]))
    # comb[:,0:features.shape[1]] = features
    # comb[:,features.shape[1]:features.shape[1]+targets.shape[1]]
    #
    # np.random.shuffle(comb)
    # subset = comb[:,]
    #kfold = KFold(n_splits=int(1.0/trainratio)) # for 80% trainingset
    #kfold = KFold(n_splits=2)
    #training_split_indexes, test_split_indexes = next(kfold.split(X, Y))
    features, targets, Y1, Y2 = makeTrainingData(X, Y, regression)
    val_features, val_targets, val_Y1, val_Y2 = makeTrainingData(o_X, o_Y, regression)
    model.compile(optimizer=optimizer["constructor"](),
                  loss='binary_crossentropy',
                  metrics=['accuracy'],loss_weights=[5,1.,1.])
    #model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.002),
    #                  metrics=['accuracy'])
    training_data = [features[:,0:X.shape[1]],features[:,X.shape[1]:2*X.shape[1]]]
    val_training_data = [val_features[:, 0:X.shape[1]], val_features[:, X.shape[1]:2 * X.shape[1]]]
    target_data = [targets,Y1,Y2]
    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = features.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    val_target_data = [val_targets, val_Y1, val_Y2]
    #tb_callback = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1,
    #                            write_graph=True, write_images=True)
    filepath = rootdir + "dualshared-weights.best.hdf5"

    run_callbacks = list()
    ret_callbacks = dict()
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_dual_ann,
                                                 datasetname,filepath,save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo

    history = model.fit(training_data, target_data,
              shuffle=True, epochs=epochs, batch_size=batch_size,
              verbose=0, callbacks=run_callbacks, validation_data=[val_training_data,val_target_data])

    return model, history, ret_callbacks


def makeEndToEndDualArchShared(o_X, o_Y, X, Y, datasetname, regression=False,
                               epochs=2000, val_split=0, shuffle=True,
                               batch_size=32, optimizer=None, onehot=True,
                               multigpu=False, callbacks=None, trainratio=0.2,
                               networklayers=[13, 13], rootdir="rootdir",alpha=0.8, makeTrainingData=None):


    input1 = Input(shape=(X.shape[1],), dtype="float32")
    input2 = Input(shape=(X.shape[1],), dtype="float32")

    if makeTrainingData is None:
        makeTrainingData = makeDualSharedArchData

    # make G(x)
    t1 = input1
    t2 = input2
    for networklayer in networklayers:
        dl1 = Dense(int(networklayer),
              activation="relu",input_shape=t1.shape)#,activity_regularizer=l2(0.01))
        #dl2 = Dense(int(networklayer),
        #      activation="relu",input_shape=t2.shape)#,activity_regularizer=l2(0.01))
        #create_shared_weights(dl,dl2,t1._keras_shape)
        t1 = dl1(t1)
        t2 = dl1(t2)
    #encoded_i1 = dl1(t1)
    #encoded_i2 = dl2(t2)

    #subtracted = Subtract()([encoded_i1,encoded_i2])

    # TODO: We had 5 layers here (from subtract to output), maybe compare
    # different "top-layers" vs (the combination) "bottom-layers", in which
    # case we need more than one layers paramater

    # make C(x,y)
    o_t = Lambda(keras_sqrt_diff)([t1, t2])
    for networklayer in networklayers:
        o_t = Dense(int(networklayer), activation="relu")(o_t)

    #make class output from G(x) to get two more signal sources
    inner_output = None
    if regression is True: # regression or 1 output classification
        inner_output1 = Dense(Y.shape[1], activation='linear',
                         kernel_initializer="random_uniform",name="reg_output1")
        inner_output2 = Dense(Y.shape[1], activation='linear',
                         kernel_initializer="random_uniform",name="reg_output2")
    else: # onehot
        inner_output1 = Dense(Y.shape[1], activation='softmax',name="class1_output")
        inner_output2 = Dense(Y.shape[1], activation='softmax',name="class2_output")
        #create_shared_weights(inner_output1,inner_output2,encoded_i1._keras_shape)

    output = Dense(1, activation="sigmoid",name="dist_output")(o_t)

    output1 = inner_output1(t1)
    output2 = inner_output2(t2)

    model = Model(inputs=[input1,input2],outputs=[output,output1,output2])

    #model.summary()
    features, targets, Y1, Y2 = makeTrainingData(X, Y, regression)
    val_features, val_targets, val_Y1, val_Y2 = makeTrainingData(o_X, o_Y, regression)

    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = features.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    if regression is not True:
        loss_dict = {'dist_output':'binary_crossentropy','class1_output':'categorical_crossentropy','class2_output':'categorical_crossentropy'}
        lossweight_dict={'dist_output':alpha,'class1_output':(1.0-alpha)/2.0,'class2_output':(1.0-alpha)/2.0}
    else:
        loss_dict = {'dist_output':'mean_squared_error','reg_output1':'mean_squared_error','reg_output2':'mean_squared_error'}
        lossweight_dict={'dist_output':alpha,'reg_output1':(1.0-alpha)/2.0,'reg_output2':(1.0-alpha)/2.0}

    model.compile(optimizer=optimizer["constructor"](),
                  loss=loss_dict,
                  metrics=['accuracy'],loss_weights=lossweight_dict)
    training_data = [features[:,0:X.shape[1]],features[:,X.shape[1]:2*X.shape[1]]]
    val_training_data = [val_features[:, 0:X.shape[1]], val_features[:, X.shape[1]:2 * X.shape[1]]]
    target_data = [targets,Y1,Y2]
    val_target_data = [val_targets, val_Y1, val_Y2]

    run_callbacks = list()
    ret_callbacks = dict()
    filepath = rootdir + "dualshared-weights.best.hdf5"
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_dual_ann,
                                                 datasetname,filepath,save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo


    # check 5 epochs
    test = np.hstack((features,targets))
    batch_size = features.shape[0]
    history = model.fit(training_data, target_data,
                        shuffle=True, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=run_callbacks,
                        validation_data=[val_training_data, val_target_data])

    return model, history, ret_callbacks


def dees_resnet(o_X, o_Y, X, Y, datasetname, regression=False,
                               epochs=2000, val_split=0, shuffle=True,
                               batch_size=32, optimizer=None, onehot=True,
                               multigpu=False, callbacks=None, trainratio=0.2,
                               networklayers=[13, 13], rootdir="rootdir",alpha=0.8, makeTrainingData=None):

    if makeTrainingData is None:
        makeTrainingData = makeDualSharedArchData

    input1 = Input(shape=(X.shape[1],), dtype="float32")
    input2 = Input(shape=(X.shape[1],), dtype="float32")

    t1 = input1
    t2 = input2
    for networklayer in networklayers:
        dl1 = Dense(int(networklayer),
              activation="relu",input_shape=t1.shape)#,activity_regularizer=l2(0.01))
        #dl2 = Dense(int(networklayer),
        #      activation="relu",input_shape=t2.shape)#,activity_regularizer=l2(0.01))
        #create_shared_weights(dl,dl2,t1._keras_shape)
        t1 = dl1(t1)
        t2 = dl1(t2)
    #encoded_i1 = dl1(t1)
    #encoded_i2 = dl2(t2)

    #subtracted = Subtract()([encoded_i1,encoded_i2])

    # TODO: We had 5 layers here (from subtract to output), maybe compare
    # different "top-layers" vs (the combination) "bottom-layers", in which
    # case we need more than one layers paramater

    subbed = Subtract()([t1,t2])
    #concatted = Concatenate()([t1,t2]) #"residual" to carry over the signals lost in the subtraction..?
    o_t = Concatenate()([t1,t2,subbed])
    #o_t = Dense(int(networklayer[0]),activation="relu")()
    for networklayer in networklayers:
        o_t = Dense(int(networklayer), activation="relu")(o_t)

    inner_output = None
    if regression is True: # regression or 1 output classification
        inner_output1 = Dense(Y.shape[1], activation='linear',
                         kernel_initializer="random_uniform",name="reg_output1")
        inner_output2 = Dense(Y.shape[1], activation='linear',
                         kernel_initializer="random_uniform",name="reg_output2")
    else: # onehot
        inner_output1 = Dense(Y.shape[1], activation='softmax',name="class1_output")
        inner_output2 = Dense(Y.shape[1], activation='softmax',name="class2_output")
        #create_shared_weights(inner_output1,inner_output2,encoded_i1._keras_shape)

    output = Dense(1, activation="sigmoid",name="dist_output")(o_t)

    output1 = inner_output1(t1)
    output2 = inner_output2(t2)

    model = Model(inputs=[input1,input2],outputs=[output,output1,output2])

    #model.summary()
    features, targets, Y1, Y2 = makeTrainingData(X, Y, regression)
    val_features, val_targets, val_Y1, val_Y2 = makeTrainingData(o_X, o_Y, regression)

    if all([optimizer is not None, optimizer["batch_size"] is not None]):
        batch_size = features.shape[0]
    else:
        batch_size = normalizeBatchSize(X, batch_size)
    if regression is not True:
        loss_dict = {'dist_output':'binary_crossentropy','class1_output':'categorical_crossentropy','class2_output':'categorical_crossentropy'}
        lossweight_dict={'dist_output':alpha,'class1_output':(1.0-alpha)/2.0,'class2_output':(1.0-alpha)/2.0}
    else:
        loss_dict = {'dist_output':'mean_squared_error','reg_output1':'mean_squared_error','reg_output2':'mean_squared_error'}
        lossweight_dict={'dist_output':alpha,'reg_output1':(1.0-alpha)/2.0,'reg_output2':(1.0-alpha)/2.0}

    model.compile(optimizer=optimizer["constructor"](),
                  loss=loss_dict,
                  metrics=['accuracy'],loss_weights=lossweight_dict)
    training_data = [features[:,0:X.shape[1]],features[:,X.shape[1]:2*X.shape[1]]]
    val_training_data = [val_features[:, 0:X.shape[1]], val_features[:, X.shape[1]:2 * X.shape[1]]]
    target_data = [targets,Y1,Y2]
    val_target_data = [val_targets, val_Y1, val_Y2]

    run_callbacks = list()
    ret_callbacks = dict()
    filepath = rootdir + "dualshared-weights.best.hdf5"
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](o_X, o_Y, X, Y,
                                                 batch_size, eval_dual_ann,
                                                 datasetname,filepath,save_best_only=True)
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo


    # check 5 epochs
    test = np.hstack((features,targets))
    batch_size = features.shape[0]
    history = model.fit(training_data, target_data,
                        shuffle=True, epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=run_callbacks,
                        validation_data=[val_training_data, val_target_data])

    return model, history, ret_callbacks
