import keras
import numpy as np
import os
from keras import backend as K


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


# from https://stackoverflow.com/questions/50127527/how-to-save-training-history-on-every-epoch-in-keras


class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self, filepath, rootdir, **kargs):
        super(CustomModelCheckPoint, self).__init__(**kargs)
        self.epoch_accuracy = {}  # loss at given epoch
        self.filepath = filepath
        self.folder = rootdir
        self.epoch_loss = {}  # accuracy at given epoch
        self.bestloss = 1
        self.lastmodelfile = ""

    def on_epoch_begin(self, epoch, logs={}):
        # Things done on beginning of epoch.
        return

    def on_epoch_end(self, epoch, logs={}):
        # things done on end of the epoch
        self.epoch_accuracy[epoch] = logs.get("acc")
        thisloss = logs.get("loss")
        self.epoch_loss[epoch] = thisloss
        if thisloss < self.bestloss:
            self.bestloss = thisloss
            # remove the last model file if it exists..
            if self.lastmodelfile is not "":
                os.remove(self.lastmodelfile)
            # save the model
            
            self.model. \
                save_weights(self.folder + "/" + self.filepath
                             + "name-of-model-%d.h5" % epoch)
            self.lastmodelfile = self.folder + \
                "/" + self.filepath + "name-of-model-%d.h5" % epoch


class RetainCallBack(keras.callbacks.Callback):
    def __init__(self, validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset, filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
        self.gvalidation_data = validation_data
        self.gvalidation_target = validation_target
        self.gtraining_data = training_data
        self.gtraining_target = training_target
        self.batch_size = batch_size
        self.eval_func = eval_func
        self.gabel=gabel

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.retain_loss = []


    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #y_pred = self.model.predict(self.model.validation_data[0])
        #self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        #if epoch > 180:
        #    print("boop")
        res = self.eval_func(self.model,self.gvalidation_data,self.gvalidation_target,self.gtraining_data,self.gtraining_target,self.batch_size)
        acc = np.sum(res) / len(res)
        self.retain_loss.append(1.0-acc)
        #print(f"epoch: {epoch} precision: {acc} loss: {logs.get('loss')}")
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

from keras import backend as K
import warnings

class Overfitting_callback(keras.callbacks.Callback):
    """Stop training when model overfits.
        # Arguments
            monitor: quantities to be monitored(list of at least two elements).
            patience: number of epochs with overfitting
                after which training will be stopped.
            verbose: verbosity mode.
            mode: one of {auto, min, max}. In `min` mode,
                training will stop when the quantity
                monitored has stopped decreasing; in `max`
                mode it will stop when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
        """

    def __init__(self,validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
        super(Overfitting_callback, self).__init__()
        monitor = ['loss', 'val_loss']
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.overfit_stop = False
        self.train_stop = False
        self.training_wait = 0
        self.overfit_wait = 0
        self.gabel=gabel

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Overfitting mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor[1]:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less


    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):

        #o1 = self.model.get_layer("dense_5").get_output_at(0).eval(session=K.get_session())
        #o2 = self.model.get_layer("dense_5").get_output_at(1).eval(session=K.get_session())
        #test = np.hstack((o1,o2))
        train_loss = logs.get(self.monitor[0])
        val_loss = logs.get(self.monitor[1])
        if val_loss is None:
            warnings.warn(
                'Overfitting conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        # train_loss > val_loss
        if not self.monitor_op(train_loss, val_loss):
            self.overfit_wait = 0

        # train_loss < val_loss
        else:
            self.overfit_wait += 1
            if self.overfit_wait >= self.patience:
                self.overfit_stop = True


        if np.less(train_loss + 0.01, self.best):
            self.best = train_loss
            self.training_wait = 0
        else:
            self.training_wait += 1
            if self.training_wait >= self.patience:
                self.train_stop = True

        if self.overfit_stop or self.train_stop:
            self.stopped_epoch = epoch
            self.model.stop_training = True


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: Model overfit. Training stopped.' % (self.stopped_epoch + 1))

class MyEarlyStop(keras.callbacks.EarlyStopping):
    def __init__(self,validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
            super(MyEarlyStop,self).__init__(monitor, min_delta, patience,
                                             verbose, mode, baseline)
            self.gabel=gabel

    def on_train_begin(self, logs=None):
        super(MyEarlyStop,self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        super(MyEarlyStop,self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(MyEarlyStop,self).on_train_end(logs)


class MyModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self,validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
            super(MyModelCheckpoint,self).__init__(monitor, min_delta, patience,
                                             verbose, mode, baseline)
            self.gabel=gabel

    def on_train_begin(self, logs=None):
        super(MyModelCheckpoint,self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        super(MyModelCheckpoint,self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(MyModelCheckpoint,self).on_train_end(logs)

class MyTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self,validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
        super(MyTensorBoard,self).__init__(log_dir, histogram_freq, write_graph, write_images)
        self.gabel=gabel

    def on_train_begin(self, logs=None):
        super(MyTensorBoard,self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        super(MyTensorBoard,self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(MyTensorBoard,self).on_train_end(logs)


class GabelStop(keras.callbacks.Callback):
    """Stop training when model exceeds gabel results
        # Arguments
            monitor: quantities to be monitored(list of at least two elements).
            patience: number of epochs with overfitting
                after which training will be stopped.
            verbose: verbosity mode.
            mode: one of {auto, min, max}. In `min` mode,
                training will stop when the quantity
                monitored has stopped decreasing; in `max`
                mode it will stop when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
        """

    def __init__(self, validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
        super(GabelStop, self).__init__()
        monitor = ['loss', 'val_loss']
        self.monitor = monitor
        self.gvalidation_data = validation_data
        self.gvalidation_target = validation_target
        self.gtraining_data = training_data
        self.gtraining_target = training_target
        self.batch_size = batch_size
        self.eval_func = eval_func
        self.dataset = dataset
        self.last_retain_loss = 1
        self.stopped_epoch = -1
        self.gabel=gabel

    def on_train_begin(self, logs=None):
        self.aucs = []
        self.losses = []
        #self.retain_loss = []

    def on_epoch_end(self, epoch, logs=None):
        res = self.eval_func(self.model,self.gvalidation_data,self.gvalidation_target,self.gtraining_data,self.gtraining_target,self.batch_size)
        acc = np.sum(res) / len(res)
        retain_loss = 1.0-acc
        gabel_retain_loss = sim_nn_results[self.dataset]
        #self.retain_loss.append(retain_loss)
        self.last_retain_loss = retain_loss
        if retain_loss < gabel_retain_loss:
            self.stopped_epoch = epoch
            self.model.stop_training = True


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: Gabel reached. Training stopped at.' % (self.stopped_epoch + 1))
            print(f"Got retain loss {self.last_retain_loss} vs gabel {sim_nn_results[self.dataset]}")

class GabelOverfit(keras.callbacks.Callback):

    def __init__(self, validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
        super(GabelOverfit, self).__init__()
        monitor = ['loss', 'val_loss']
        self.monitor = monitor
        self.gvalidation_data = validation_data
        self.gvalidation_target = validation_target
        self.gtraining_data = training_data
        self.gtraining_target = training_target
        self.batch_size = batch_size
        self.eval_func = eval_func
        self.dataset = dataset
        self.last_retain_loss = 1
        self.stopped_epoch = -1
        self.best = np.inf
        self.gabel=gabel

    def on_train_begin(self, logs=None):
        self.aucs = []
        self.losses = []
        #self.retain_loss = []

    def on_epoch_end(self, epoch, logs=None):
        res = self.eval_func(self.model,self.gvalidation_data,self.gvalidation_target,self.gtraining_data,self.gtraining_target,self.batch_size)
        acc = np.sum(res) / len(res)
        retain_loss = 1.0-acc
        gabel_retain_loss = sim_nn_results[self.dataset]
        #self.retain_loss.append(retain_loss)
        self.last_retain_loss = retain_loss
        if self.best > retain_loss:
            self.best = retain_loss
            print(f"new best: {self.best}")

        if retain_loss < gabel_retain_loss:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        if self.best != np.inf and self.best < self.last_retain_loss:
            print(f"overfit best: {self.best}")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: Gabel reached. Training stopped at.' % (self.stopped_epoch + 1))
            print(f"Got retain loss {self.last_retain_loss} vs gabel {sim_nn_results[self.dataset]}")

class GabelElitism(keras.callbacks.Callback):

    def __init__(self, validation_data, validation_target,
                 training_data, training_target, batch_size,
                 eval_func, dataset,filepath,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 save_best_only = False, save_weights_only = False,
                 period = 1, gabel=False, log_dir="./graph",
                 histogram_freq=0, write_graph=True,
                 write_images=True, normal=False, evals=False):
        super(GabelElitism, self).__init__()
        monitor = ['loss', 'val_loss']
        self.monitor = monitor
        self.gvalidation_data = validation_data
        self.gvalidation_target = validation_target
        self.gtraining_data = training_data
        self.gtraining_target = training_target
        self.batch_size = batch_size
        self.eval_func = eval_func
        self.dataset = dataset
        self.last_retain_loss = 1
        self.stopped_epoch = -1
        self.best = np.inf
        self.gabel=gabel

    def on_train_begin(self, logs=None):
        self.aucs = []
        self.losses = []
        #self.retain_loss = []

    def on_epoch_end(self, epoch, logs=None):
        res = self.eval_func(self.model,self.gvalidation_data,self.gvalidation_target,self.gtraining_data,self.gtraining_target,self.batch_size)
        acc = np.sum(res) / len(res)
        retain_loss = 1.0-acc
        gabel_retain_loss = sim_nn_results[self.dataset]
        #self.retain_loss.append(retain_loss)
        self.last_retain_loss = retain_loss
        if self.best > retain_loss:
            self.best = retain_loss


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: Gabel reached. Training stopped at.' % (self.stopped_epoch + 1))
            print(f"Got retain loss {self.last_retain_loss} vs gabel {sim_nn_results[self.dataset]}")

callbackdict = {
    "gabelstop": {"callback": GabelStop},
    "gabeloverfit": {"callback": GabelOverfit},
    "gabelelitism": {"callback": GabelElitism},
    "retain_measure": {"callback": RetainCallBack},
    "overfitting": {"callback": Overfitting_callback},
    "modelcheckpoint": {"callback": MyModelCheckpoint},
    "earlystopping": {"callback": MyEarlyStop},
    "tensorboard": {"callback": MyTensorBoard},
    "mycustomcheckpoint": {"callback": CustomModelCheckPoint}
}
