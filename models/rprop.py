from keras.optimizers import Optimizer
from keras import backend as K
import numpy

__name__ = "rprop"
"""
Both of these implementations are fixed versions of code found on
https://stackoverflow.com/questions/43768411/implementing-the-rprop-algorithm-in-keras/45849212#45849212
So credits go to the stackoverflow community and the specific members that authored the questions and answers.

"""
class RProp(Optimizer):
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        alphas = [K.variable(numpy.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for param, grad, old_grad, prev_weight_delta, alpha in zip(params, grads,
                                                                   old_grads, prev_weight_deltas,
                                                                   alphas):
            # equation 4
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0), K.maximum(alpha * self.scale_down, self.min_alpha), alpha)
            )

            # equation 5
            new_delta = K.switch(K.greater(grad, 0),
                                 -new_alpha,
                                 K.switch(K.less(grad, 0),
                                          new_alpha,
                                          K.zeros_like(new_alpha)))

            # equation 7
            weight_delta = K.switch(K.less(grad*old_grad, 0), -prev_weight_delta, new_delta)

            # equation 6
            new_param = param + weight_delta

            # reset gradient_{t-1} to 0 if gradient sign changed (so that we do
            # not "double punish", see paragraph after equation 7)
            grad = K.switch(K.less(grad*old_grad, 0), K.zeros_like(grad), grad)


            # Apply constraints
            #if param in constraints:
            #    c = constraints[param]
            #    new_param = c(new_param)

            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(prev_weight_delta, weight_delta))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class iRprop_(Optimizer):
    def __init__(self, init_alpha=0.01, scale_up=1.2, scale_down=0.5, min_alpha=0.00001, max_alpha=50., **kwargs):
        super(iRprop_, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for p, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            grad = K.sign(grad)
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0),K.maximum(alpha * self.scale_down, self.min_alpha),alpha)
            )

            grad = K.switch(K.less(grad * old_grad, 0),K.zeros_like(grad),grad)
            new_p = p - grad * new_alpha

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))

        return self.updates

    def get_config(self):
        config = {
        'init_alpha': float(K.get_value(self.init_alpha)),
        'scale_up': float(K.get_value(self.scale_up)),
        'scale_down': float(K.get_value(self.scale_down)),
        'min_alpha': float(K.get_value(self.min_alpha)),
        'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRprop_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
