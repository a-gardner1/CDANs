# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 11:28:39 2016

@author: Drew
"""

import numpy as np
import os
from keras.models import model_from_json, Model, Sequential
from keras.layers import merge, Input
from keras.layers.core import Lambda, Masking, Reshape, Dense, Flatten, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K, activations, regularizers, initializations
from keras.engine.topology import Layer, InputSpec
from keras.regularizers import Regularizer
from queryUser import queryUser
import sys
import marshal
import types as python_types
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.subtensor import take
import theano.tensor as T
import warnings

"""
Note: K.arange does not exist prior to Keras v1.2.0.

TODO: Make compatible with latest version of Keras (2.0+).
      Make compatible with Tensorflow backend (if possible).
"""

def addMaxoutLayer(model, layer, numPieces):
    """
    Adds an layer with a maxout wrapper on the output. 
    Maxout computes the piecewise maximum of multiple functions (possibly linear).
    The first (non-batch) dimension of the output of the layer must be divisible by
    numPieces, the number of functions whose maximum is being taken.

    Masking not supported.

    See http://arxiv.org/pdf/1302.4389.pdf. 
    """
    model = layer(model)
    if numPieces <= 0:
        raise ValueError("The number of pieces must be positive.")
    if numPieces > 1: #otherwise, just do normal layer
        if model._keras_shape[1] % numPieces != 0:
            raise ValueError("The output_shape of the given layer must be divisible by numPieces.")
        model = Reshape((numPieces, model._keras_shape[1]/numPieces)+ model._keras_shape[2:])(model)
        #use a mask-eating lambda instead of normal lambda since normal lambda is bugged in current version of Keras (1.2.1)
        model = MaskEatingLambda(lambda x: K.max(x, axis=1), output_shape = lambda input_shape: (input_shape[0],) + input_shape[2:])(model)
    return model

def Maxout(input_shape, layer, numPieces):
    """
    Given a layer whose output_shape is divisible by numPieces, return a wrapper
    that computes the maximum of every 'numPieces' consecutive outputs. The wrapper
    may be treated as a normal layer, i.e. model = Maxout(Dense(), k)(model).

    Masking not supported. Layers with multiple inputs not supported.
    """
    if numPieces > 1:
        inputLayer = Input(input_shape)
        maxout = addMaxoutLayer(inputLayer, layer, numPieces)
        maxout = Model(inputLayer, maxout)
        return maxout
    else:
        return layer

def addResidualLayer(model, layer, identityMapping=True, **kwargs):
    """
    Add a residual connection between the model's current output layer and the output after the given
    layer is added to the model.

    If the output_shape does not change after adding the layer and the identityMapping argument is True,
    then the identity mapping is used.
    Otherwise, a linear projection is used and expected to be learned during training.

    Masking not supported. See addDenseResidualLayers().

    See https://arxiv.org/pdf/1512.03385.pdf.
    """
    return addDenseResidualLayers(model, [layer], identityMapping, **kwargs)

def Residual(input_shape, layer, identityMapping=True, **kwargs):
    return DenseResidual(input_shape, [layer], identityMapping, **kwargs)

def addDenseResidualLayers(model, layers, identityMappings=[True], **kwargs):
    """
    Add a series of layers such that there is a residual feed-forward connection between each pair of 
    layers that are to be added to the given model.

    For n layers, results in n(n-1)/2 residual connections.
    
    If the output_shape does not change after adding a layer and the identityMapping argument is True,
    then the identity mapping is used.
    Otherwise, a linear projection is used and expected to be learned during training.

    If a given layer has a Dropout object as its final sublayer, then the residual connection is incorporated
    in such a way that it will share the same Dropout object.

    Masking is not supported, nor are layers with multiple inputs or outputs.

    TODO: Make general function that takes adjacency list between layers to determine residual connections.

    See https://arxiv.org/abs/1608.06993.
    """
    if isinstance(identityMappings, bool):
        identityMappings = [identityMappings]
    try:
        identityMappings = list(identityMappings)
    except:
        raise AttributeError("identityMappings must be boolean or iterable.")
    if len(identityMappings) == 1:
        identityMappings = [identityMappings[0] for layer in layers]
    elif len(identityMappings) != len(layers):
        raise ValueError("'identityMappings' must be the same length as 'layers'")
    #keep a list of intermediate outputs for each layer of the model
    intermediateModels = [model]
    #model will keep track of the topmost output including all residual connections added so far
    for layer in layers:
        #try to transparently remove Dropout layer
        dropoutLayer = None
        if isinstance(layer.layers[-1], Dropout):
            dropoutLayer = layer.layers[-1]
            #should not be possible for Dropout to be preceded by multiple inbound nodes
            assert(len(layer.nodes_by_depth[1]) == 1)
            layer = Model(layer.input, layer.nodes_by_depth[1][0].output_tensors) 
        normalModel = layer(model)
        #add residual connection to each previous layer
        residualModels = []
        for residualModel, identityMapping in zip(intermediateModels, identityMappings):
            if residualModel._keras_shape != normalModel._keras_shape or not identityMapping:
                reshape = False
                if residualModel.ndim >= 3:
                    reshape = True
                    residualModel = Flatten()(residualModel)
                residualModel = Dense(np.prod(normalModel._keras_shape[1:]), activation = 'linear', **kwargs)(residualModel)
                if reshape:
                    residualModel = Reshape(normalModel._keras_shape[1:])(residualModel)
            residualModels += [residualModel]
        model = merge(residualModels + [normalModel], mode = 'sum')
        #reapply Dropout to sum if appropriate
        if dropoutLayer is not None:
            model = dropoutLayer(model)
        intermediateModels += [model]
    return model

def DenseResidual(input_shape, layers, identityMappings=[True], **kwargs):
    inputLayer = Input(input_shape)
    residual = addDenseResidualLayers(inputLayer, layers, identityMappings, **kwargs)
    residual = Model(inputLayer, residual)
    return residual

def addPermutationalLayer(model, f, pooling_mode = 'ave'):
    """
    Add (the equivalent of) a permutational layer to the given model, which provides
    a permutation equivariant output based upon pairwise combinations of the input.

    Certain pairs are pooled to limit the size of the output and are controlled by the 
    'pooling_mode' parameter, which is one of 'ave', 'sum', 'max', or 'min'.

    The provided function 'f' is applied to each pairwise combination and should be a Keras model
    or layer that supports masking, provided that the given model has masked output.

    See http://arxiv.org/1612.04530.
    """
    input_shape = model._keras_shape
    model = Pairwise()(model)
    model = f(model)
    model = MaskedPooling(pool_size = input_shape[1], stride = input_shape[1], mode = pooling_mode)(model)
    return model

def addRealGaussianNoise(model, sigma, masking, maskValue = 0):
    """
    Add a layer to a model that adds Gaussian noise with standard deviation sigma 
    and ignores timesteps that would normally be masked even if no mask is present.
    """
    if not masking: #need to ensure that padded zeros do not get corrupted by noise
        model = Masking(maskValue)(model)
    model = GaussianNoise(sigma)(model)
    if not masking:
        model = MaskEatingLambda(lambda x, mask: K.switch(K.expand_dims(mask, -1), x, maskValue), 
                                 lambda input_shape: input_shape)(model)
    return model

class MeanWeightRegularizer(Regularizer):
    """
    Implements the version of l2 regularization that I was unwittingly using
    with Keras 1.0.6. This uses the mean instead of the sum and is therefore
    not technically l2 regularization but it is similar. Requires a significantly
    different range of values for l1 and l2. 
    Pros: Values are mostly model independent.
    Cons: Effects probably not as well understood as actual l1/l2 regularization.
    """
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'MeanWeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a MeanWeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="ml2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss + K.mean(K.abs(self.p)) * self.l1
        regularized_loss += K.mean(K.square(self.p)) * self.l2
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}

def ml2(l=0.01):
    return MeanWeightRegularizer(l2=l)

class PermutationEquivariant(Layer):
    """
    Implements the permutation equivariant layer of Ravanbahksh et al. for use with sets and point clouds.

    Specifically, implements Equation 5,

    \sigma(\beta + (x-\mathbf{1}\transpose{\max_n x})\Gamma),

    where x is n by k, \max_n x is K by 1, \mathbf{1} is n by 1, \beta is k^\prime by 1, and \Gamma is k by k^\prime.
    \beta and \Gamma are the trainable parameters

    Currently supports a built-in version of maxout with k pieces, specified by the keyword maxout = k.

    See http://arxiv.org/1611.04500 (version 3)
    """
    def __init__(self, output_dim, init = 'glorot_uniform', activation = None, 
                 Gamma_regularizer = None, beta_regularizer = None,
                 maxout = 1, **kwargs):
        self.init = initializations.get(init)
        self.supports_masking = True
        self.uses_learning_phase = True
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.maxout = maxout
        self.Gamma_regularizer = regularizers.get(Gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        return super(PermutationEquivariant, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        #output channel weights
        self.Gamma = self.add_weight(name = 'Gamma',
                                     shape = (input_shape[2], self.output_dim * self.maxout),
                                     initializer = self.init,
                                     regularizer = self.Gamma_regularizer,
                                     trainable = True)
        #bias
        self.beta = self.add_weight(name = 'beta',
                                     shape = (self.output_dim * self.maxout, ),
                                     initializer = self.init,
                                     regularizer = self.beta_regularizer,
                                     trainable = True)
        super(PermutationEquivariant, self).build(input_shape)
        
    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1], self.output_dim)

    def compute_mask(self, input, input_mask = None):
        return input_mask

    def call(self, x, mask = None):
        if mask is None:
            output = x - K.repeat(TimeDistributedMerge(False, mode = 'max')(x), self.input_spec[0].shape[1])
        else:
            output = x - K.repeat(TimeDistributedMerge(True, mode = 'max')(x, mask), self.input_spec[0].shape[1])
        output = K.dot(output, self.Gamma)
        output = self.beta + output
        if self.activation is not None:
            output = self.activation(output)
        if self.maxout > 1:
            output = K.max(K.reshape(output, (-1, self.input_spec[0].shape[1], self.output_dim, self.maxout)), axis = -1)
        return output

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'init': self.init.__name__,
            'activation': self.activation.__name__,
            'maxout': self.maxout,
            'Gamma_regularizer': self.Gamma_regularizer.get_config() if self.Gamma_regularizer else None,
            'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
        }
        base_config = super(PermutationEquivariant, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Pairwise(Layer):
    """
    Compute all pairwise combinations of timesteps (axis 1).
    A pairwise combination is a concatenation of the features of one timestep with those of another.
    If the input has shape (None, N, M, ...), returns shape (None, N*N, 2*M, ...).

    TODO: Add support for other axes.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = False
        return super(Pairwise, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if len(input_shape) < 3:
            raise ValueError("The number of dimensions of each sample must be greater than or equal to 2")
        super(Pairwise, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1]*input_shape[1], 2*input_shape[2]) + input_shape[3:]
    
    def compute_mask(self, x, input_mask = None):
        """
        Treat each sample's mask as a vector and compute outer product with itself.
        Then flatten the resulting outer product.
        """
        if input_mask is not None:
            return K.reshape(K.expand_dims(input_mask, 1)*K.expand_dims(input_mask, -1), 
                             (input_mask.shape[0], input_mask.shape[1]*input_mask.shape[1]))
        else:
            return None

    def call(self, x, mask = None):
        """
        Compute all pairwise combinations of timesteps, ignoring the mask.
        """
        input_shape = self.input_spec[0].shape
        x2 = K.repeat_elements(x, input_shape[1], axis=1)
        x2 = K.reshape(x2, (-1, input_shape[1], input_shape[1])+input_shape[2:])
        x3 = K.permute_dimensions(x2, [0, 2, 1] + range(3, K.ndim(x2)))
        x2 = K.reshape(x2, (x.shape[0], -1) + input_shape[2:])
        x3 = K.reshape(x3, (x.shape[0], -1) + input_shape[2:])
        x4 = K.concatenate([x2, x3], axis = 2)
        return x4

class Reverse(Layer):
    """
    Reverse the given tensor along the specified axis. 
    If a mask is given and reverseMask is True, the mask is also reversed along the same axis.
    Note that since masks only work along the time axis (axis 1),
    any other given axis has no practical effect on the mask.
    Not compatible with Tensorflow backend.
    """
    def __init__(self, axis = 1, reverseMask = True, **kwargs):
        if not isinstance(axis, (int, long)):
            warnings.warn('Attempting to cast provided axis to integer.')
            axis = int(axis)
        if axis == 0:
            raise ValueError('Cannot reverse the batch dimension (axis 0).')
        elif axis < 0:
            raise ValueError('Provided axis must be a positive integer.')
        self.axis = axis
        self.reverseMask = reverseMask
        self.supports_masking = True
        self.uses_learning_phase = False
        super(Reverse, self).__init__(**kwargs)

    def compute_mask(self, x, input_mask = None):
        if input_mask is not None:
            if self.axis == 1 and self.reverseMask: #masks are per timestep (dimension 1)
                return take(input_mask, T.arange(input_mask.shape[self.axis]-1, -1, -1), self.axis)
            else: #therefore, reversing other dimensions has no effect on the mask
                return input_mask
        else:
            return None

    def call(self, x, mask = None):
        return take(x, T.arange(x.shape[self.axis]-1, -1, -1), self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Reverse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskedPooling(Layer):
    """
    The built-in pooling functions of Keras (as of 4/3/2017) do not appear
    to support masking. Input is implicitly padded with masked zeros.

    Pools exclusively along the time dimension for now.
    Not compatible with Tensorflow backend.
    """
    def __init__(self, pool_size = 2, stride = 1, mode = 'max', **kwargs):
        modes = ['sum', 'ave', 'max', 'min']
        if pool_size >= 2:
            self.pool_size = pool_size
        else:
            raise ValueError("'pool_size' must be greater than or equal to 2.")
        if stride >= 1:
            self.stride = stride
        else:
            raise ValueError("'stride' must be greater than or equal to 1.")
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError('Illegal pooling mode provided: ' + str(mode) + '.')
        self.supports_masking = True
        self.uses_learning_phase = False
        super(MaskedPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(MaskedPooling, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], (input_shape[1]-1)/self.stride+1) + input_shape[2:]

    def compute_mask(self, input, input_mask = None):
        if input_mask is not None:
            input_shape = self.input_spec[0].shape
            # If the pool contains at least one unmasked input (nonzero mask), 
            # then the output is unmasked (nonzero).
            numPools = (input_shape[1]-1)/self.stride+1
            if numPools == 1:
                output_mask = K.expand_dims(K.sum(input_mask[:, T.arange(0, K.minimum(self.pool_size, input_shape[1]))], axis = 1), -1)
            else:
                poolStarts = K.expand_dims(K.expand_dims(T.arange(0, input_shape[1], self.stride), -1), 0)
                _, pools, _ = K.rnn(lambda p, t: (K.expand_dims(K.sum(t[-1][:, T.arange(p[0,0],K.minimum(p[0,0]+self.pool_size,input_shape[1]))], axis = 1), 0), []),
                                    poolStarts, [], False, None, [input_mask], unroll = True, input_length = numPools)
                output_mask = pools
            #for poolStart in poolStarts:
            #    pools += [K.sum(input_mask[:,poolStart:(poolStart+self.pool_size)], axis = 1)]
            #output_mask = K.concatenate(pools, axis = 1)
            return K.switch(output_mask, 1, 0)
        else:
            return None

    def call(self, x, mask = None):
        input_shape = self.input_spec[0].shape
        x = K.reshape(x, (x.shape[0], x.shape[1], -1))
        numPools = (input_shape[1]-1)/self.stride+1
        poolStarts = K.expand_dims(K.expand_dims(T.arange(0, input_shape[1], self.stride), -1), 0)
        if mask is not None:
            _, pools, _ = K.rnn(lambda p, t: (K.expand_dims(TimeDistributedMerge(True, self.mode)(
                                                              t[-2][:, T.arange(p[0,0],K.minimum(p[0,0]+self.pool_size,input_shape[1])), :],
                                                              t[-1][:, T.arange(p[0,0],K.minimum(p[0,0]+self.pool_size,input_shape[1]))]
                                                              ), 
                                                           0), []),
                                poolStarts, [], False, None, [x, mask], unroll = True, input_length = numPools)
        else:
            _, pools, _ = K.rnn(lambda p, t: (K.expand_dims(TimeDistributedMerge(False, self.mode)(
                                                              t[-1][:, T.arange(p[0,0],K.minimum(p[0,0]+self.pool_size,input_shape[1])), :]
                                                              ), 
                                                           0), []),
                                poolStarts, [], False, None, [x], unroll = True, input_length = numPools)
        if numPools == 1:
            pools = K.expand_dims(pools, -1)
        output = pools
        #if mask is not None:
        #    # If the pool contains at least one unmasked input (nonzero mask), 
        #    # then the output is unmasked (nonzero).
        #    masks = []
        #    for poolStart in poolStarts:
        #        masks += [mask[:,poolStart:(poolStart+self.pool_size)]]
        #else:
        #    masks = [K.cast_to_floatx(1) for poolStart in poolStarts]
        #pools = []
        #for poolStart, poolmask in zip(poolStarts, masks):
        #    pool = x[:, poolStart:(poolStart+self.pool_size), :]
        #    pools += [TimeDistributedMerge(mask is not None, self.mode)(pool, poolmask)]
        #output = K.concatenate(pools, axis = 1)
        return K.reshape(output, (x.shape[0], numPools)+input_shape[2:])
        
    def get_config(self):
        config = {'stride': self.stride,
                  'pool_size': self.pool_size,
                  'mode': self.mode}
        base_config = super(MaskedPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Sort(Layer):
    """
    Sort the input along the second to last axis according to the 
    sorted order of input[:,:,:,0]. 
    For sorting 3D marker positions by their X value.
    Support for other axes and sorting by lambdas may be added in the future.
    Masked inputs are treated as though they are larger than any unmasked inputs,
    although this only applies if the input is a 3D array.
    Not compatible with the TensorFlow backend.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = False
        super(Sort, self).__init__(**kwargs)
        
    def compute_mask(self, x, input_mask = None):
        if input_mask is not None:
            if K.ndim(x) == 3: #masks are per timestep (dimension 1)
                #all masked values are moved to the end
                #based on http://stackoverflow.com/questions/43010969/numpy-matrix-move-all-0s-to-the-end-of-each-row
                flipped_mask = T.gt(K.sum(input_mask, axis = 1, keepdims = 1), T.arange(input_mask.shape[1]-1, -1, -1))
                flipped_mask = take(flipped_mask, T.arange(flipped_mask.shape[1]-1, -1, -1), 1)
                input_mask = T.set_subtensor(input_mask[flipped_mask.nonzero()], input_mask[input_mask.nonzero()])
                input_mask = T.set_subtensor(input_mask[K.switch(flipped_mask, 0, 1).nonzero()], 0)
                return input_mask
            else: #therefore, sorting other dimensions has no effect on the mask
                return input_mask
        else:
            return None

    def call(self, x, mask = None):
        if mask is not None and K.ndim(x) == 3:
            #replace masked values with large values
            modX = K.max(x)+K.cast_to_floatx(1)
            modX = K.switch(K.expand_dims(mask, -1), x, modX)
            indices = T.argsort(modX, axis = -2)
        else: #mask has no effect even if it exists
            indices = T.argsort(x, axis = -2)
        # don't know how to do this without reshaping
        input_shape = x.shape
        indices = K.reshape(take(indices, 0, -1), (-1, input_shape[-2]))
        x = K.reshape(x, (-1, input_shape[-2], input_shape[-1]))
        x = x[K.expand_dims(T.arange(input_shape[0]), -1), indices]
        return K.reshape(x, input_shape)

    def get_config(self):
        return super(Sort, self).get_config()

def func_dump(func):
    """
    Adapted from Keras fix #2814 (can be found in generic_utils in a later release). 
    Didn't want to upgrade at this time (1.0.7) in case of compatibility issues.
    Serializes a lambda function.
    """
    py3 = sys.version_info[0] == 3
    if py3:
        code = marshal.dumps(func.__code__).replace(b'\\',b'/').decode('raw_unicode_escape')
        defaults = func.__defaults__
        if func.__closure__:
            closure = tuple([c.cell_contents for c in func.__closure__])
        else:
            closure = None
    else:
        code = marshal.dumps(func.func_code).replace(b'\\',b'/').decode('raw_unicode_escape')
        defaults = func.func_defaults
        if func.func_closure:
            closure = tuple([c.cell_contents for c in func.func_closure])
        else:
            closure = None
    return code, defaults, closure

def func_load(code, defaults=None, closure=None, globs=None):
    """
    Adapted from Keras fix #2814 (can be found in generic_utils in a later release). 
    Didn't want to upgrade at this time (1.0.7) in case of compatibility issues.
    Deserializes a lambda function.
    """
    py3 = sys.version_info[0] == 3
    if isinstance(code, (tuple, list)):  # unpack previous dump
        code, defaults, closure = code
    code = marshal.loads(code.encode('raw_unicode_escape'))
    if closure is not None:
        closure = func_reconstruct_closure(closure)
    if globs is None:
        globs = globals()
    return python_types.FunctionType(code, globs, name=code.co_name, argdefs=defaults, closure=closure)

def func_reconstruct_closure(values):
    '''Deserialization helper that reconstructs a closure.'''
    nums = range(len(values))
    src = ["def func(arg):"]
    src += ["  _%d = arg[%d]" % (n, n) for n in nums]
    src += ["  return lambda:(%s)" % ','.join(["_%d" % n for n in nums]), ""]
    src = '\n'.join(src)
    try:
        exec(src)
    except:
        raise SyntaxError(src)
    py3 = sys.version_info[0] == 3
    return func(values).__closure__ if py3 else func(values).func_closure

class MaskEatingLambda(Layer):
    """
     Saw references to this class, but cannot find it anywhere.
     Try to recreate. 
     Removes a mask from the pipeline while performing a custom operation.
     Takes two functions as arguments. The first, 'function', computes the output of 
     the layer given the input and an optional mask (see definition of call).
     The second, 'output_shape', computes the shape of the output as a function of 
     the shape of the input. If not provided, the input_shape is the default value.
    """
    def __init__(self, function, output_shape = None, **kwargs):
        self.function = function
        self.supports_masking = True
        self.uses_learning_phase = False
        self._output_shape = output_shape
        super(MaskEatingLambda, self).__init__(**kwargs)

    # do not return a mask. Eat it.
    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        if mask is None:
            return self.function(x)
        else:
            return self.function(x, mask)
    
    #from Keras source for normal Lambda layer
    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            # if TensorFlow, we can infer the output shape directly:
            if K._BACKEND == 'tensorflow':
                if type(input_shape) is list:
                    xs = [K.placeholder(shape=shape) for shape in input_shape]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape)
                    x = self.call(x)
                if type(x) is list:
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # otherwise, we default to the input shape
            return input_shape
        elif type(self._output_shape) in {tuple, list}:
            nb_samples = input_shape[0] if input_shape else None
            return (nb_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if type(shape) not in {list, tuple}:
                raise Exception('output_shape function must return a tuple')
            return tuple(shape)
            
    def get_config(self):
        if isinstance(self.function, python_types.LambdaType):
            function = func_dump(self.function)
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            output_shape = func_dump(self._output_shape)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type}
        base_config = super(MaskEatingLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        function_type = config.pop('function_type')
        if function_type == 'function':
            function = globals()[config['function']]
        elif function_type == 'lambda':
            function = func_load(config['function'], globs = globals())
        else:
            raise Exception('Unknown function type: ' + function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = func_load(config['output_shape'], globs = globals())
        else:
            output_shape = config['output_shape']

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)

def TimeDistributedMerge(isMasked = False, mode = 'ave'):
    """
    Merge along the time-axis (dimension 1, 0-indexed). Takes a mask to 
    determine which time-samples to ignore in the merge. 

    This effectively performs a sort of pooling operation depending on the mode.
    """
    modes = ['sum', 'ave', 'max', 'min', 'concat']
    if mode in modes:
        if isMasked:
            if mode == 'ave':
                # make so that when the mask is all 0, the output is 0.
                def ave(x, mask):
                    sums = K.cast(K.sum(mask, axis=-1, keepdims=True), 'float32')
                    sums = K.switch(sums, sums, K.ones_like(sums))
                    return K.batch_dot(x,mask, axes=1) / sums
                return ave
                #return lambda x, mask: K.batch_dot(x,mask, axes=1) / K.cast(K.sum(mask, axis=-1, keepdims=True), 'float32')
            elif mode =='concat':
                return lambda x, mask: K.batch_flatten(x) #ignores mask so that output is fixed size
            elif mode == 'sum':
                return lambda x, mask: K.batch_dot(x, mask, axes=1)
            elif mode == 'max':
                def max(x, mask):
                    #wherever the input is masked, set it equal to the minimum
                    mins = K.min(x, axis=1, keepdims = True)
                    # expand mask (replace zeros in x with 1 so that 0 = masked)
                    mask = K.expand_dims(mask)*K.switch(x, x, K.ones_like(x))
                    mins = K.switch(mask, x, mins) # replace values
                    #now masked values have no effect on the maximum
                    return K.max(mins, axis=1)
                return max
            else:
                def min(x, mask):
                    #wherever the input is masked, set it equal to the maximum
                    maxes = K.max(x, axis=1, keepdims = True)
                    # expand mask (replace zeros in x with 1 so that 0 = masked)
                    mask = K.expand_dims(mask)*K.switch(x, x, K.ones_like(x))
                    maxes = K.switch(mask, x, maxes)#replace values
                    #now masked values have no effect on the minimum
                    return K.min(maxes, axis=1)
                return min
        else:
            if mode == 'ave':
                return lambda x: K.mean(x, axis=1)
            elif mode == 'concat':
                return lambda x: K.batch_flatten(x)
            elif mode == 'sum':
                return lambda x: K.cast(K.sum(x, axis=1), 'float32')
            elif mode == 'max':
                return lambda x: K.max(x, axis=1)
            else:
                return lambda x: K.min(x, axis=1)
    else:
        raise ValueError('Illegal merge mode provided:' + str(mode))
        
   
         
class SimultaneousDropout(Layer):
    """
    Applies Dropout to the input on a temporal basis. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    Simultaneous dropout is an extension of this idea which drops random features across all timesteps.
    # Arguments
        p: float between 0 and 1. Fraction of the features to drop.
        rescale: boolean. Scale remaining timesteps by the percentage dropped. 
                 If there were n timesteps and r were dropped, multiplies remaining
                 timesteps by n/(n-r). 
                 If r=n, does nothing (activation is zero regardless).
    # References
        - [Deep Unordered Composition Rivals Syntactic Methods for Text Classification]
    """
    def __init__(self, p,  seed=None, **kwargs):
        self.p = p
        self.seed = seed
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(SimultaneousDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape = input_shape)]
        return super(SimultaneousDropout, self).build(input_shape)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            noise_shape = (x.shape[0], 1) + self.input_spec[0].shape[2:]

            def dropped_inputs():
                return K.dropout(x, self.p, noise_shape, seed=self.seed)
            x = K.in_train_phase(dropped_inputs, lambda: x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(SimultaneousDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
         
class WordDropout(Layer):
    """
    Applies Dropout to the input on a temporal basis. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    Word dropout is an extension of this idea which drops random timesteps or 
    word embeddings, especially for use with deep averaging networks.
    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
        rescale: boolean. Scale remaining timesteps by the percentage dropped. 
                 If there were n timesteps and r were dropped, multiplies remaining
                 timesteps by n/(n-r). 
                 If r=n, does nothing (activation is zero regardless).
    # References
        - [Deep Unordered Composition Rivals Syntactic Methods for Text Classification]
    """
    def __init__(self, p, rescale, **kwargs):
        self.p = p
        self.rescale = rescale 
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(WordDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # adapt normal dropout code (Theano only)
        def wdropout(x, level, mask=None, seed=None, rescale=False):
            if level < 0. or level >= 1:
                raise Exception('Dropout level must be in interval [0, 1[.')
            if seed is None:
                seed = np.random.randint(1, 10e6)
            rng = RandomStreams(seed=seed)
            retain_prob = 1. - level
            dropoutMask = K.expand_dims(rng.binomial((x.shape[0],x.shape[1]), p=retain_prob, dtype=x.dtype))
            x *= dropoutMask
            #x /= retain_prob # this rescaling is part of the dropout algorithm,
            # but we aren't exactly doing real dropout with the same purpose in mind;
            # therefore...don't do this? 
            if rescale:
                # scale so that overall activation per sample is maintained
                # first calculate n-r
                # now calculate n
                if mask is not None:
                    #restrict to dropped real outputs
                    emask = K.expand_dims(mask)
                    scale = K.cast(K.sum(dropoutMask*emask, axis=1, keepdims = True), 'float32')
                    scale /= K.cast(K.sum(emask, axis=1, keepdims=True), 'float32')
                else:
                    scale = K.cast(K.sum(dropoutMask, axis=1, keepdims = True), 'float32')
                    scale /= x.shape[1]
                #avoid division by 0. If not zero divide. Otherwise do nothing.
                x = K.switch(scale, x/scale, x)
            return x
        if 0. < self.p < 1.:
            x = K.in_train_phase(wdropout(x, level=self.p, mask=mask), x)
        return x
        

    def get_config(self):
        config = {'p': self.p, 'rescale': self.rescale}
        base_config = super(WordDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def accuracy(y_true, y_pred, binaryClassMatrix=False, sample_mask=None):
    """
    Compute the accuracy given true and predicted class labels.
    
    If 'binaryClassMatrix' is true, then both y_true and y_pred are matrices
    of shape (numSamples, ..., numClasses). A binary array 'sample_mask' must
    also be provided to determine which samples to ignore as filler data, where
    a weight of 0 (or False) indicates filler data.
    """
    if binaryClassMatrix:
        if sample_mask is not None:
            y_true = np.argmax(y_true, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true[sample_mask == 0] = -1
        else:
            raise ValueError('Sample masks must be provided  for each target if binaryClassMatrix is True.')
    return float((y_true == y_pred).sum())/float((y_true>=0).sum())
    
def balancedAccuracy(y_true, y_pred, binaryClassMatrix=False, sample_mask=None):
    """
    Compute the accuracy given true and predicted class labels. Normalizes each
    sample to have the same weight.
    
    If 'binaryClassMatrix' is true, then both y_true and y_pred are matrices
    of shape (numSamples, ..., numClasses). A binary array 'sample_mask' must
    also be provided to determine which samples to ignore as filler data, where
    a weight of 0 (or False) indicates filler data.
    """
    if binaryClassMatrix:
        if sample_mask is not None:
            y_true = np.argmax(y_true, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true[sample_mask == 0] = -1
        else:
            raise ValueError('Sample weights must be provided for each target if binaryClassMatrix is True.')
    weights = (y_true>=0).sum(axis=1)
    return np.mean(((y_true == y_pred).T.astype('float32')/weights).sum(axis=0))
    
def weightedAccuracy(y_true, y_pred, binaryClassMatrix=False, sample_mask=None,
                     forgetFactor = 0, initialWeight = 0.01):
    """
    The last term in each sequence is given a weight of 1. Prior terms are 
    weighted by their successors' weight multiplied by the forget factor.
    If no forgetFactor is provided, then it is automatically determined for 
    each sequence by the initialWeight, which is the weight given to the first
    term in each sequence.
    
    By default (with a forget factor of 0), the accuracy only considers the 
    final frame of each sequence and thus serves as a measure of how well the
    network eventually gets the right answer.
    """
    if binaryClassMatrix:
        if sample_mask is not None:
            y_true = np.argmax(y_true, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true[sample_mask == 0] = -1
        else:
            raise ValueError('Sample weights must be provided for each target if binaryClassMatrix is True.')
    lengths = (y_true>=0).sum(axis=1)
    if forgetFactor is None:
        weights = [np.concatenate((np.power(np.power(initialWeight, 1.0/length), np.array(range(length-1, -1, -1))),
                                   np.zeros((y_true.shape[-1]-length,)))) for length in lengths]
        weights = np.asarray(weights)
    else:
        weights = [np.concatenate((np.power(forgetFactor, np.array(range(length-1, -1, -1))),
                                   np.zeros((y_true.shape[-1]-length,)))) for length in lengths]
        weights = np.asarray(weights)
    return ((y_true == y_pred).T.astype('float32')*weights.T).sum()/weights.sum()
    
def weightSamplesByTimeDecay(y_true, binaryClassMatrix=False, sample_mask=None,
                     forgetFactor = 0, initialWeight = 0.01):
    """
    The last term in each sequence is given a weight of 1. Prior terms are 
    weighted by their successors' weight multiplied by the forget factor.
    If no forgetFactor is provided, then it is automatically determined for 
    each sequence by the initialWeight, which is the weight given to the first
    term in each sequence.
    
    By default (with a forget factor of 0), the accuracy only considers the 
    final frame of each sequence and thus serves as a measure of how well the
    network eventually gets the right answer.
    """
    if abs(forgetFactor) > 1:
        raise ValueError("Argument 'forgetFactor' must be a number in the range [0,1].")
    if binaryClassMatrix:
        if sample_mask is not None:
            y_true = np.argmax(y_true, axis=-1)
            y_true[sample_mask == 0] = -1
        else:
            raise ValueError('Sample weights must be provided for each target if binaryClassMatrix is True.')
    lengths = (y_true>=0).sum(axis=1)
    if forgetFactor is None:
        weights = [np.concatenate((np.power(np.power(initialWeight, 1.0/length), np.array(range(length-1, -1, -1))),
                                   np.zeros((y_true.shape[-1]-length,)))) for length in lengths]
        weights = np.asarray(weights)
    else:
        weights = [np.concatenate((np.power(forgetFactor, np.array(range(length-1, -1, -1))),
                                   np.zeros((y_true.shape[-1]-length,)))) for length in lengths]
        weights = np.asarray(weights)
    return weights

def prepareData(sequences, classes, numClasses, sampleWeights):
    numSamples = [sequence.shape[1] for sequence in sequences]
    if np.unique(numSamples).shape[0] > 1:
        raise ValueError('Each input must have the same number of samples. Received inputs with sample counts ' + str(numSamples))
    isTemporal = classes.shape[0] > 1
    isTemporal2 = [sequence.shape[0] > 1 for sequence in sequences]
    #class labels must be made into binary arrays
    binaryClasses = np.zeros((classes.shape[0], classes.shape[1], numClasses))
    # tell cost function which timesteps to ignore
    if sampleWeights is None:
        calcSampleWeights = True
        sampleWeights = np.ones((classes.shape[0], classes.shape[1]))
    else:
        calcSampleWeights = False
    #eh...just use for loops
    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
            if classes[i,j] >= 0:
                binaryClasses[i,j, classes[i,j]] = 1
            elif calcSampleWeights:
                sampleWeights[i,j] = 0
    #range over samples (sequences) in first dimension, time in second, features in third
    sequences = [sequence.transpose([1,0] + range(2, len(sequence.shape))) for sequence in sequences]
    binaryClasses = binaryClasses.transpose((1,0,2))
    if not isTemporal:
        binaryClasses = binaryClasses.reshape((binaryClasses.shape[0], binaryClasses.shape[2]))
    if calcSampleWeights:
        sampleWeights = sampleWeights.T
    return isTemporal, isTemporal2, sequences, binaryClasses, sampleWeights

def getModelOutput(model):
    """
    Utility function for examining the training output of a model.
    Assumes the model has only one output.
    """
    O = K.function(model.inputs + [K.learning_phase()], model.output)
    return lambda x: O(x, True)

def getModelGradients(model):
    """
    Utility function for examining the gradients during training.
    Returns functions capable of computing gradients with respect to different parts of the network.

    To get i-th gradient: GF[i].__call__(x, y, sample_weights, 1)
    """
    #get symbolic representations of the gradient updates
    #updates = model.optimizer.get_updates(collect_trainable_weights(model), model.constraints, model.total_loss)
    grads = model.optimizer.get_gradients(model.total_loss, collect_trainable_weights(model))
    GF = [K.function(model.inputs + model.targets + model.sample_weights + [K.learning_phase()], grad) for grad in grads]
    return GF


def trainKerasModel(model, batchSize,
                    numEpochs,  
                    sequences, classes, trainRange, 
                    valRange, testRange,
                    numClasses, modelFile, 
                    callbacks = None, sampleWeights = None, 
                    outDirectory = '', trainMode = 'continue',
                    custom_objects = {},
                    loss_function = 'categorical_crossentropy',
                    optimizer = RMSprop(0.001)):
    """
    Returns True if training was completed, False if interrupted.
    
     # sequences:       List of arrays of sequences. 
                        Each array of sequences must be the same length (i.e. same number of samples = shape[1]).
                        This allows for multiple inputs.
     # classes:         List of target values. Integers. Multiple outputs not currently supported.
     # custom_objects:  Dictionary of name-object pairs for custom layers and 
                        such used in the provided model. 
                        Example:
                        
                        custom_objects = {'MaskEatingLambda': MaskEatingLambda}
    """
    trainModes = ['continue', 'overwrite', 'skip']
    
    if trainMode.lower() not in trainModes:
        raise ValueError("Parameter 'trainMode' must be one of 'continue', 'overwrite', or 'skip'")
        
    if outDirectory is not None and outDirectory != '':
        outDirectory = outDirectory + '\\'
    else:
        outDirectory = ''

    isTemporal, isTemporal2, sequences, binaryClasses, sampleWeights = prepareData(sequences, classes, numClasses, sampleWeights)
    
    trainData = [[sequence[trainRange,:,:] if isTemp2 else sequence[trainRange,:].squeeze() for sequence, isTemp2 in zip(sequences, isTemporal2)], 
                 binaryClasses[trainRange,:,:] if isTemporal else binaryClasses[trainRange,:], 
                 sampleWeights[trainRange, :]]
    valData = [[sequence[valRange,:,:] if isTemp2 else sequence[valRange,0,:].squeeze() for sequence, isTemp2 in zip(sequences, isTemporal2)], 
               binaryClasses[valRange,:,:] if isTemporal else binaryClasses[valRange,:], 
               sampleWeights[valRange, :]]
    testData = [[sequence[testRange,:,:] if isTemp2 else sequence[testRange,0,:].squeeze() for sequence, isTemp2 in zip(sequences, isTemporal2)], 
                binaryClasses[testRange, :, :] if isTemporal else binaryClasses[testRange,:], 
                sampleWeights[testRange, :]]
        
    modelFile = outDirectory + 'Keras'+modelFile
    weightsFile = modelFile+'_Weights'
    completedEpochs = 0
    # if a pre-trained model exists and we are not set to overwrite, load it.
    # otherwise, we use the provided model
    if not ((trainMode == 'overwrite') 
            or (not os.path.isfile(modelFile+'.json') 
            or not os.path.isfile(weightsFile+'.h5'))):
        model = model_from_json(open(modelFile+'.json', 'rb').read(), custom_objects)
        model.load_weights(weightsFile+'.h5')
    
    #compile model and training objective function
    #sgd = SGD(lr=learningRate)
    #adagrad = Adagrad(lr=learningRate)
    model.compile(loss=loss_function, optimizer=optimizer,
                  sample_weight_mode='temporal' if isTemporal else 'none', 
                  metrics=['accuracy'])
    checkp = [ModelCheckpoint(weightsFile + '.h5', save_best_only = True)]
    if callbacks is None:
        callbacks = checkp
    else:
        callbacks += checkp
    try:
        if trainMode != 'skip':
            completedEpochs = model.fit(x=trainData[0], y=trainData[1], 
                                        sample_weight=trainData[2] if isTemporal else None,
                                        validation_data = valData if isTemporal else (valData[0], valData[1]), 
                                        batch_size = batchSize, 
                                        nb_epoch = numEpochs, callbacks = callbacks,
                                        verbose = 2)
            completedEpochs = completedEpochs.history
            #completedEpochs = completedEpochs.history['loss']
    except KeyboardInterrupt:
        if(not queryUser('Training interrupted. Compute test statistics?')):
            if isTemporal:
                return 0, float('nan'), float('nan'), float('nan') 
            else:
                return 0, float('nan'), float('nan')
    #retrieve the best weights based upon validation set loss
    if os.path.isfile(weightsFile+'.h5'):
        model.load_weights(weightsFile+'.h5')
    scores = model.test_on_batch(x=testData[0], y=testData[1], 
                                 sample_weight=testData[2] if isTemporal else None)
    predictedClasses = predict_classes(model, testData[0], isTemporal)
    if not isTemporal:
        predictedClasses = predictedClasses.reshape((predictedClasses.shape[0], 1))
    scores[1] = accuracy(classes[:, testRange].T, predictedClasses)
    if not isTemporal:
        print("Test loss of %.5f\nAccuracy of %.5f" % (scores[0], scores[1]))
    else:
        scores.append(balancedAccuracy(classes[:, testRange].T, predictedClasses))
        scores.append(weightedAccuracy(classes[:, testRange].T, predictedClasses, forgetFactor=0))
        print("Test loss of %.5f\nFrame-wise accuracy of %.5f\nSequence-wise accuracy of %.5f\nFinal frame accuracy of %0.5f" % (scores[0], scores[1], scores[2], scores[3]))
    if trainMode != 'skip':
        modelString = model.to_json()
        open(modelFile + '.json', 'wb').write(modelString)
        model.save_weights(weightsFile + '.h5', overwrite=True)
        print('Model and weights saved to %s and %s.' % (modelFile+'.json', weightsFile+'.h5'))

    if isTemporal:
        return completedEpochs, scores[0], scores[1], scores[2], scores[3]
    else:
        return completedEpochs, scores[0], scores[1]

def evaluateEnsemble(modelDicts, 
                     sequences, classes, 
                     testRange, numClasses, sampleWeights):
    """
    Takes a list of dictionaries (composed of parameters to trainKerasModel()) 
    that define models that will be combined into an ensemble by simple averaging
    of their output layers. If the model does not yet exist (indicated by a 'None'
    in the model given in the modelDict), then it is loaded from the modelFile 
    given in modelDicts.
    If the model exists but is not trained (indicated by modelFile not existing), 
    then it is trained via trainKerasModel(). 

    If both the model and modelFile exist, then parameters in the corresponding
    modelDicts entry can be ignored. The only required entries are 'outDirectory',
    'modelFile', and 'model'.
    """
    models = []
    for modelDict in modelDicts:
        #does the model exist?
        outDirectory = modelDict['outDirectory'] 
        modelFile = ('\\Keras' if outDirectory != '' else 'Keras') + modelDict['modelFile']
        if modelDict['model'] is None:
            if (os.path.isfile(modelFile+'.json') 
                and os.path.isfile(modelFile+'_Weights.h5')):
                model = model_from_json(open(modelFile+'.json', 'rb').read(), modelDict['custom_objects'])
                model.load_weights(modelFile+'_Weights.h5')
            else:
                raise ValueError('If the model is None, then the modelFile must exist (i.e. the model must be pretrained)!'
                                 +'The architecture of the model is not known here.')
        elif (not os.path.isfile(modelFile+'.json') 
                or not os.path.isfile(modelFile+'_Weights.h5')):
            #does the modelFile exist?
            trainKerasModel(modelDict['model'], modelDict['batchSize'],
                            modelDict['numEpochs'], modelDict['learningRate'],
                            modelDict['sequences'], modelDict['classes'], 
                            modelDict['trainRange'], modelDict['valRange'], modelDict['testRange'],
                            numClasses, modelDict['modelFile'], modelDict['callbacks'],
                            modelDict['sampleWeights'], modelDict['outDirectory'],
                            'continue', modelDict['custom_objects'])
            model = model_from_json(open(modelFile+'.json', 'rb').read(), modelDict['custom_objects'])
            model.load_weights(modelFile+'_Weights.h5')
        else:
            model = modelDict['model']
        models += [model]
    #ensemble parts loaded. evaluate it on the given test set
    #first prepare the test set
    isTemporal, isTemporal2, sequences, binaryClasses, sampleWeights = prepareData(sequences, classes, numClasses, sampleWeights)
    
    testData = [sequences[testRange, :, :] if isTemporal2 else sequences[testRange,:].squeeze(), 
                binaryClasses[testRange, :, :] if isTemporal else binaryClasses[testRange,:], 
                sampleWeights[testRange, :]]
    #make actual ensemble model and compile
    inputLayer = Input(shape = testData[0].shape[1:])
    if len(models) > 1:
        outputLayer = merge([model(inputLayer) for model in models], mode = 'ave')
    else:
        outputLayer = models[0](inputLayer)
    ensemble = Model(input = inputLayer, output = outputLayer)
    ensemble.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  sample_weight_mode='temporal' if isTemporal else 'none', 
                  metrics=['accuracy'])
    #test
    scores = ensemble.test_on_batch(x=testData[0], y=testData[1], 
                                    sample_weight=testData[2] if isTemporal else None)
    predictedClasses = predict_classes(ensemble, testData[0], isTemporal)
    scores[1] = accuracy(classes[:, testRange].T, predictedClasses)
    if not isTemporal:
        print("Test loss of %.5f\nAccuracy of %.5f" % (scores[0], scores[1]))
    else:
        scores.append(balancedAccuracy(classes[:, testRange].T, predictedClasses))
        scores.append(weightedAccuracy(classes[:, testRange].T, predictedClasses, forgetFactor=0))
        print("Test loss of %.5f\nFrame-wise accuracy of %.5f\nSequence-wise accuracy of %.5f\nFinal frame accuracy of %0.5f" % (scores[0], scores[1], scores[2], scores[3]))
    
def predict_classes(model, x, isTemporal = False):
    if isinstance(model, Sequential):
        return model.predict_classes(x)
    elif isinstance(model, Model):
        probs = model.predict(x)
        if probs.shape[-1] > 1:
            predictedClasses = probs.argmax(axis=-1)
        else:
            predictedClasses = (probs > 0.5).astype('int32')
        if not isTemporal:
            predictedClasses = predictedClasses.reshape((predictedClasses.shape[0], 1))
        return predictedClasses
    else:
        raise ValueError("Unknown type: 'model' must be a Keras 'Model' or 'Sequential' object.")

"""
Potentially could use this instead of my hack by changing supports_masking to True.
From Keras source code.
class Lambda(Layer):
    '''Used for evaluating an arbitrary Theano / TensorFlow expression
    on the output of the previous layer.
    # Examples
    ```python
        # add a x -> x^2 layer
        model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
        # add a layer that returns the concatenation
        # of the positive part of the input and
        # the opposite of the negative part
        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)
        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)
        model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
    ```
    # Arguments
        function: The function to be evaluated.
            Takes one argument: the output of previous layer
        output_shape: Expected output shape from function.
            Can be a tuple or function.
            If a tuple, it only specifies the first dimension onward; 
                 sample dimension is assumed either the same as the input:
                 `output_shape = (input_shape[0], ) + output_shape`
                 or, the input is `None` and the sample dimension is also `None`:
                 `output_shape = (None, ) + output_shape`
            If a function, it specifies the entire shape as a function of 
                 the input shape: `output_shape = f(input_shape)`
        arguments: optional dictionary of keyword arguments to be passed
            to the function.
    # Input shape
        Arbitrary. Use the keyword argument input_shape
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Specified by `output_shape` argument.
    '''
    def __init__(self, function, output_shape=None, arguments={}, **kwargs):
        self.function = function
        self.arguments = arguments
        self.supports_masking = False

        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            if not hasattr(output_shape, '__call__'):
                raise Exception('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape
        super(Lambda, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            # if TensorFlow, we can infer the output shape directly:
            if K._BACKEND == 'tensorflow':
                if type(input_shape) is list:
                    xs = [K.placeholder(shape=shape) for shape in input_shape]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape)
                    x = self.call(x)
                if type(x) is list:
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # otherwise, we default to the input shape
            return input_shape
        elif type(self._output_shape) in {tuple, list}:
            nb_samples = input_shape[0] if input_shape else None
            return (nb_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if type(shape) not in {list, tuple}:
                raise Exception('output_shape function must return a tuple')
            return tuple(shape)

    def call(self, x, mask=None):
        arguments = self.arguments
        arg_spec = inspect.getargspec(self.function)
        if 'mask' in arg_spec.args:
            arguments['mask'] = mask
        return self.function(x, **arguments)

    def get_config(self):
        py3 = sys.version_info[0] == 3

        if isinstance(self.function, python_types.LambdaType):
            if py3:
                function = marshal.dumps(self.function.__code__).decode('raw_unicode_escape')
            else:
                function = marshal.dumps(self.function.func_code).decode('raw_unicode_escape')
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            if py3:
                output_shape = marshal.dumps(self._output_shape.__code__).decode('raw_unicode_escape')
            else:
                output_shape = marshal.dumps(self._output_shape.func_code).decode('raw_unicode_escape')
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type,
                  'arguments': self.arguments}
        base_config = super(Lambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        function_type = config.pop('function_type')
        if function_type == 'function':
            function = globals()[config['function']]
        elif function_type == 'lambda':
            function = marshal.loads(config['function'].encode('raw_unicode_escape'))
            function = python_types.FunctionType(function, globals())
        else:
            raise Exception('Unknown function type: ' + function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = marshal.loads(config['output_shape'].encode('raw_unicode_escape'))
            output_shape = python_types.FunctionType(output_shape, globals())
        else:
            output_shape = config['output_shape']

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)
"""