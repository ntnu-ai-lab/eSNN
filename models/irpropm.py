
from keras.optimizers import Optimizer
from keras import backend as K
import numpy as np
#import theano
import tensorflow as tf

__name__ = "irpropm"

def shared_zeros(shape, dtype=tf.float32, name='', n=None):
    shape = shape if n is None else (n,) + tuple(shape)
    return tf.Variable(tf.zeros(shape, dtype=dtype), name=name)

def sharedX(X, dtype=tf.float32, name=None):
    return tf.Variable(np.asarray(X, dtype=dtype), name=name)

class iRPROPm(Optimizer):
    '''
        Reference: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332
    '''
    def __init__(self, step_inc=1.2, step_dec=0.5, step_init=0.001, step_min=1e-7, step_max=50.0, *args, **kwargs):
        super(iRPROPm, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.step_inc = step_inc
        self.step_dec = step_dec
        self.step_min = step_min
        self.step_max = step_max
        self.step_init = step_init

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        c=0
        prev_grads = [shared_zeros(p.shape,name=f"prev_grads") for p in params]
        c=0
        prev_steps = [sharedX(np.full(p.shape, self.step_init),name=f"prev_stepd") for p in params]
        self.updates = []

        for p, grad, prev_grad, prev_step, c in zip(params, grads, prev_grads,
                                   prev_steps):

            grad_sgn = prev_grad * grad

            new_step = K.switch(K.ge(grad_sgn, 0.0),
                                K.minimum(prev_step * self.step_inc, self.step_max),
                                K.maximum(prev_step * self.step_dec, self.step_min))
            
            self.updates.append((prev_step, new_step))
            
            new_grad = K.switch(K.ge(grad_sgn, 0.0), grad, 0.0)
            self.updates.append((prev_grad, new_grad))
            
            new_p = p - K.sgn(new_grad) * new_step
            self.updates.append((p, c(new_p)))
            
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "step_inc": float(self.step_inc),
                "step_dec": float(self.step_dec),
                "step_min": float(self.step_min),
                "step_max": float(self.step_max),
                "step_init": float(self.step_init)}
