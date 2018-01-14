# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 10:15:00 2016

@author: Drew
"""

from keras.models import Sequential, Model
from keras.layers import merge, Input
from keras.layers.core import Merge, Dense, Masking, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop
from Gestures import loadDataset, normalizePercentages
import numpy as np
import sys, os, time, warnings
from KerasSupplementary import trainKerasModel, weightSamplesByTimeDecay, addRealGaussianNoise
from KerasSupplementary import MaskEatingLambda, TimeDistributedMerge, WordDropout, Reverse, Sort, MaskedPooling
from KerasSupplementary import addMaxoutLayer, addResidualLayer, addDenseResidualLayers, Maxout, Residual, DenseResidual
from KerasSupplementary import addPermutationalLayer, PermutationEquivariant, SimultaneousDropout, Pairwise

def defaultDirectory():
    return '.'

def getCustomObjects():
    return {'MaskEatingLambda': MaskEatingLambda, 'WordDropout': WordDropout, 'Reverse': Reverse, 
            'Sort': Sort, 'MaskedPooling': MaskedPooling, 'Pairwise': Pairwise, 
            'PermutationEquivariant': PermutationEquivariant, 'SimultaneousDropout': SimultaneousDropout}

def getMask(data, mask_value = 0, keepdims = True):
    """
    Return the numpy array equivalent of a Keras Masking layer.
    """
    mask = np.any(np.not_equal(data.reshape((data.shape[0], data.shape[1],-1)), mask_value), 
                  axis = -1, keepdims = keepdims)
    if keepdims:
        mask = (mask*np.ones_like(data.reshape((data.shape[0], data.shape[1],-1)), dtype = 'bool')).reshape(data.shape)
    return mask

def centerData(data, mask = False, prependMean = False):
    """
    Shift the origin of each frame to be the mean of its unmasked positions.
    If prependMean is True, then place the mean in the first index of each frame.
    """
    if mask:
        mask = getMask(data, keepdims = True)
    else:
        mask = np.ones_like(data, dtype = 'bool')
    mean = np.sum(data, axis = 1, keepdims = True)
    counts = np.sum(mask, axis = 1, keepdims = True)
    fullMean = np.zeros_like(data) + mean/counts #restore full dimensions
    fullMean[~mask] = 0
    data = data-fullMean
    if prependMean:
        data = np.concatenate([mean, data], axis = 1)
    return data

def normalizeData(data, mask = False):
    """
    Normalize each coordinate axis to lie between the range of 0 and 1.
    Try to reproduce Ademola's work.
    """
    if mask:
        mask = getMask(data, keepdims = True)
    else:
        mask = np.ones_like(data, dtype = 'bool')
    max = np.max(data[mask])
    min = np.min(data[mask])
    data[~mask] = min
    maxes = np.max(data, axis = 1, keepdims = True)
    data[~mask] = max
    mins = np.min(data, axis = 1, keepdims = True)
    data = (data-mins)/(maxes-mins)
    data[~mask] = 0
    return data


def sortData(data, W, mask = False):
    """
    Given actual (not symbolic) input data of shape (numSamples, ..., maxNumMarkers, 3)
    and a rank 3 weight matrix W, sorts the second to last axis according to the rows
    of W, which represent linearly independent normal vectors of planes.
    Points are lexicographically sorted based on their perpendicular distance to each 
    plane in the row order of W, 
    i.e. they are sorted by the first normal, then ties are resolved by
    the second, and ties after this sort are again resolved by the final normal.
    """
    if mask:
        mask = getMask(data, keepdims = True)
    else:
        mask = np.ones_like(data, dtype == 'bool')
    W = W[::-1, :].T #place primary normal vector in last column
    scores = np.dot(data, W).reshape((-1, data.shape[-2], data.shape[-1]))
    #masked values are pushed to the end
    scores[mask == 0] = np.max(scores)
    orders = np.asarray([np.lexsort(score.T) for score in scores], dtype = 'int')
    sortedData = np.asarray([sample[order] for sample, order in zip(data.reshape((-1, data.shape[-2], data.shape[-1])), orders)])
    return sortedData.reshape(data.shape)

def makePostureDeepCNNModel(input, numConvLayers,
                            numClasses, numFilters = 11, 
                            activation = 'relu',
                            poolSize = 2, l2Reg = 0.0001, 
                            dropoutH = 0, poolingMode = 'max',
                            stripSoftmax = False,
                            numDeepLayers = 1, deepNetSize = 11,
                            masking = True):
    """
    Due to the limitations of Keras, we cannot make a variable size
    convolutional neural network (with a variable number of layers)
    without a bunch of extra programming. Therefore, please note that the
    number of convolutional layers should ideally be set to 
    log_{poolSize}(max input size).

    Note that this is not quite a normal convolutional neural network.

    Inspired by [Multi-column Deep Neural Networks for Image Classification] by Ciresan et al.
    """
    model = input
    # each pass through the loop drops the number of time steps by a factor of poolSize
    # and increases the number of filters by numFilters
    for l in xrange(numConvLayers):
        model = TimeDistributed(Dense(output_dim=numFilters*(l+1), activation=activation, 
                                        W_regularizer=l2(l2Reg)))(model)
        if dropoutH:
            model = Dropout(dropoutH)(model)
        model = TimeDistributed(Dense(output_dim=numFilters*(l+1), activation=activation, 
                                        W_regularizer=l2(l2Reg)))(model)
        if dropoutH:
            model = Dropout(dropoutH)(model)
        model = MaskedPooling(pool_size = poolSize, stride = poolSize, 
                              mode = poolingMode)(model)
    model = TimeDistributed(Dense(output_dim=numFilters*(numConvLayers+1), activation=activation, 
                                    W_regularizer=l2(l2Reg)))(model)
    if dropoutH:
        model = Dropout(dropoutH)(model)
    model = MaskEatingLambda(TimeDistributedMerge(isMasked = masking, mode = poolingMode), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(model)
    for i in range(numDeepLayers-1):
        model = Dense(output_dim=deepNetSize, activation=activation, 
                        W_regularizer = l2(l2Reg))(model)
        if dropoutH:
            model = Dropout(dropoutH)(model)
    model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                    W_regularizer=l2(l2Reg))(model)
    return model


def makePostureRCNNModel(input, numClasses,
                         rnnHiddenSize = 11, deepNetSize = 11,
                         numRNNLayers = 1, numDeepLayers = 3,
                         bidirectional=False, l2Reg = 0.0001,
                         activation='relu', useGRU=False,
                         dropoutH = 0,
                         makeSymmetric = False, stripSoftmax = False,
                         poolingMode = 'max', poolSize = 2, poolStride = 2,
                         masking = True):
    """
    Make a recurrent deep averaging network using an LSTM. 
    Might be easier to follow if Keras' functional API is used (done!).
    """
    model = input
    if bidirectional:
        #def fork(mod, n=2):
        #    #obsolete function
        #    forks = []
        #    for i in range(n):
        #        f = Sequential()
        #        f.add(mod)
        #        forks.append(f)
        #    return forks
        rnnSize = rnnHiddenSize
        if makeSymmetric:
            if useGRU:
                forward = GRU(output_dim=rnnSize, return_sequences=True, 
                              W_regularizer=l2(l2Reg))
            else:
                forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                               W_regularizer=l2(l2Reg))
            backward = forward(Reverse()(model))
            forward = forward(model)
        else:
            if useGRU:
                forward = GRU(output_dim=rnnSize, return_sequences=True, 
                                W_regularizer=l2(l2Reg))(model)
                backward = GRU(output_dim=rnnSize, return_sequences=True, 
                                W_regularizer=l2(l2Reg))(Reverse()(model))
            else:
                forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                 W_regularizer=l2(l2Reg))(model)
                backward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                  W_regularizer=l2(l2Reg))(Reverse()(model))
        for i in range(numRNNLayers-1):
            # combine forward and backward LSTMs
            model = merge([forward, Reverse()(backward)], mode = 'concat')
            if dropoutH:
                model = Dropout(dropoutH)(model)
            model = MaskedPooling(pool_size = poolSize, stride = poolSize, 
                                  mode = poolingMode)(model)
            # split outputs so that the next forward-backward layer can be made
            rnnSize = (i+2)*rnnHiddenSize
            if makeSymmetric:
                if useGRU:
                    forward = GRU(output_dim=rnnSize, return_sequences=True, 
                                  W_regularizer=l2(l2Reg))
                else:
                    forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                   W_regularizer=l2(l2Reg))
                backward = forward(Reverse()(model))
                forward = forward(model)
            else:
                if useGRU:
                    forward = GRU(output_dim=rnnSize, return_sequences=True, 
                                    W_regularizer=l2(l2Reg))(model)
                    backward = GRU(output_dim=rnnSize, return_sequences=True, 
                                    W_regularizer=l2(l2Reg))(Reverse()(model))
                else:
                    forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                     W_regularizer=l2(l2Reg))
                    backward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                      W_regularizer=l2(l2Reg))(Reverse()(model))
        #combine forward and backward for unidirectional feedforward layers
        model = merge([forward, Reverse()(backward)], mode = 'concat')
        if dropoutH:
            model = Dropout(dropoutH)(model)
        model = MaskedPooling(pool_size = poolSize, stride = poolSize, 
                                mode = poolingMode)(model)
    else:
        for i in range(numRNNLayers):
            rnnSize = (i+1)*rnnHiddenSize
            if useGRU:
                model = GRU(output_dim=rnnSize, return_sequences=True, 
                            W_regularizer=l2(l2Reg))(model)
            else:
                model = LSTM(output_dim=rnnSize, return_sequences=True, 
                             W_regularizer=l2(l2Reg))(model)
            if dropoutH:
                model = Dropout(dropoutH)(model)
            model = MaskedPooling(pool_size = poolSize, stride = poolSize, 
                                  mode = poolingMode)(model)
    # deep averaging network
    model = MaskEatingLambda(TimeDistributedMerge(isMasked = masking, mode = poolingMode), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(model)
    for i in range(numDeepLayers-1):
        model = Dense(output_dim=deepNetSize, activation=activation, 
                      W_regularizer=l2(l2Reg))(model)
        if dropoutH:
            model = Dropout(dropoutH)(model)
    model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                    W_regularizer=l2(l2Reg))(model)
    return model

def makeFPMEQModel(input, numClasses, fHiddenSize = 11,
                  fOutSize = 11, deepNetSize = 11,
                  fNumLayers = 2, numDeepLayers = 3,
                  activation = 'relu', l2Reg = 0.0001,
                  dropoutH = 0, stripSoftmax = False, 
                  poolingMode = 'ave', masking = True,
                  maxout = 1, deepResidual = False,
                  batchSize = 64):
    """
    Make a fixed structure permutation equivariant (FPMEQ) model.

    Simultaneous dropout is applied to the permutation equivariant layers
    
    See http://arxiv.org/1611.04500 (version 3, equation 5).
    """
    try:
        maxout = list(maxout)
    except:
        maxout = [maxout, maxout]
    deepMaxout = maxout[1]
    maxout = maxout[0]
    model = input
    #make permutation equivariant function layer by layer
    for i in range(fNumLayers):
        output_dim =(fHiddenSize if i < (fNumLayers-1) else fOutSize)
        model = PermutationEquivariant(output_dim=output_dim, activation=activation, 
                                       Gamma_regularizer=l2(l2Reg), maxout = maxout)(model)
        if dropoutH:
            model = SimultaneousDropout(dropoutH)(model)
    #if fResidual == 'dense':
    #    f = DenseResidual((input._keras_shape[2]*2,), layers, False, **{'W_regularizer': l2(l2Reg)})
    #else:
    #    fInput = Input((input._keras_shape[2]*2,))
    #    f = fInput
    #    for layer in layers:
    #        f = layer(f)
    #    f = Model(fInput, f)
    #    if fResidual:
    #        f = Residual((input._keras_shape[2],), f, True, **{'W_regularizer': l2(l2Reg)})
    #deep averaging network
    model = MaskEatingLambda(TimeDistributedMerge(isMasked = masking, mode = poolingMode), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(model)
    layers = []
    for i in range(numDeepLayers-1):
        layerInput = Input((fOutSize if i == 0 else deepNetSize,))
        layer = Maxout(layerInput._keras_shape[1:],
                       Dense(output_dim=deepNetSize*deepMaxout, activation=activation, 
                             W_regularizer = l2(l2Reg)), deepMaxout)(layerInput)
        if dropoutH:
            layer = Dropout(dropoutH)(layer)
        layer = Model(layerInput, layer)
        layers += [layer]
    if deepResidual == 'dense':
        mlp = DenseResidual((fOutSize,), layers, False, **{'W_regularizer': l2(l2Reg)})
    else:
        mlpInput = Input((fOutSize,))
        mlp = mlpInput
        for layer in layers:
            mlp = layer(mlp)
        mlp = Model(mlpInput, mlp)
        if deepResidual:
            mlp = Residual((fOutSize,), mlp, True, **{'W_regularizer': l2(l2Reg)})
    model = mlp(model)
    model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                    W_regularizer=l2(l2Reg))(model)
    return model

def makePMEQModel(input, numClasses, fHiddenSize = 11,
                  fOutSize = 11, deepNetSize = 11,
                  fNumLayers = 2, numDeepLayers = 3,
                  activation = 'relu', l2Reg = 0.0001,
                  dropoutH = 0, stripSoftmax = False, 
                  poolingMode = 'ave', masking = True,
                  maxout = 1, fResidual = False, deepResidual = False):
    """
    Make a permutation equivariant (PMEQ) model.
    
    See http://arxiv.org/1612.04530.
    """
    if not (isinstance(poolingMode, list) or isinstance(poolingMode, tuple)):
        poolingMode = [poolingMode, poolingMode]
    fPoolingMode = poolingMode[0]
    poolingMode = poolingMode[1]
    try:
        maxout = list(maxout)
    except:
        maxout = [maxout, maxout]
    deepMaxout = maxout[1]
    maxout = maxout[0]
    model = input
    layers = []
    #make pairwise function layer by layer
    for i in range(fNumLayers):
        layerInput = Input((input._keras_shape[2]*2 if i == 0 else fHiddenSize,))
        layer = Maxout(layerInput._keras_shape[1:],
                       Dense(output_dim=(fHiddenSize if i < (fNumLayers-1) else fOutSize)*maxout, 
                             activation=activation, 
                             W_regularizer=l2(l2Reg)), maxout)(layerInput)
        if dropoutH:
            layer = Dropout(dropoutH)(layer)
        layer = Model(layerInput, layer)
        layers += [layer]
    if fResidual == 'dense':
        f = DenseResidual((input._keras_shape[2]*2,), layers, False, **{'W_regularizer': l2(l2Reg)})
    else:
        fInput = Input((input._keras_shape[2]*2,))
        f = fInput
        for layer in layers:
            f = layer(f)
        f = Model(fInput, f)
        if fResidual:
            f = Residual((input._keras_shape[2],), f, True, **{'W_regularizer': l2(l2Reg)})
    model = addPermutationalLayer(model, TimeDistributed(f), fPoolingMode)
    #deep averaging network
    model = MaskEatingLambda(TimeDistributedMerge(isMasked = masking, mode = poolingMode), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(model)
    layers = []
    for i in range(numDeepLayers-1):
        layerInput = Input((fOutSize if i == 0 else deepNetSize,))
        layer = Maxout(layerInput._keras_shape[1:],
                       Dense(output_dim=deepNetSize*deepMaxout, activation=activation, 
                             W_regularizer = l2(l2Reg)), deepMaxout)(layerInput)
        if dropoutH:
            layer = Dropout(dropoutH)(layer)
        layer = Model(layerInput, layer)
        layers += [layer]
    if deepResidual == 'dense':
        mlp = DenseResidual((fOutSize,), layers, False, **{'W_regularizer': l2(l2Reg)})
    else:
        mlpInput = Input((fOutSize,))
        mlp = mlpInput
        for layer in layers:
            mlp = layer(mlp)
        mlp = Model(mlpInput, mlp)
        if deepResidual:
            mlp = Residual((fOutSize,), mlp, True, **{'W_regularizer': l2(l2Reg)})
    model = mlp(model)
    model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                    W_regularizer=l2(l2Reg))(model)
    return model

def makePostureCDANModel(input, numClasses, convNetHiddenSize = 11, 
                         convNetOutSize = 11, deepNetSize = 11,
                         numCNNLayers = 2, numDeepLayers = 3,
                         activation='relu', l2Reg = 0.0001,
                         dropoutH = 0,
                         stripSoftmax=False, poolingMode = 'ave',
                         masking = True, maxout = 1, 
                         convNetResidual=False, deepResidual=False):
    """
    Make a convolutional deep averaging network.
    """
    try:
        maxout = list(maxout)
    except:
        maxout = [maxout, maxout]
    deepMaxout = maxout[1]
    maxout = maxout[0]
    model = input
    layers = []
    #make embedding function layer by layer
    for i in range(numCNNLayers):
        layerInput = Input((input._keras_shape[2] if i == 0 else convNetHiddenSize,))
        layer = Maxout(layerInput._keras_shape[1:],
                       Dense(output_dim=(convNetHiddenSize if i < (numCNNLayers-1) else convNetOutSize)*maxout, 
                             activation=activation, 
                             W_regularizer=l2(l2Reg)), maxout)(layerInput)
        if dropoutH:
            layer = Dropout(dropoutH)(layer)
        layer = Model(layerInput, layer)
        layers += [layer]
    #combine embedding layers into one function
    if convNetResidual == 'dense':
        embedding = DenseResidual((input._keras_shape[2],), layers, False, **{'W_regularizer': l2(l2Reg)})
    else:
        embeddingInput = Input((input._keras_shape[2],))
        embedding = embeddingInput
        for layer in layers:
            embedding = layer(embedding)
        embedding = Model(embeddingInput, embedding)
        if convNetResidual:
            embedding = Residual((input._keras_shape[2],), embedding, True, **{'W_regularizer': l2(l2Reg)})
    model = TimeDistributed(embedding)(model)
    #deep averaging network
    model = MaskEatingLambda(TimeDistributedMerge(isMasked = masking, mode = poolingMode), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(model)
    layers = []
    for i in range(numDeepLayers-1):
        layerInput = Input((convNetOutSize if i == 0 else deepNetSize,))
        layer = Maxout(layerInput._keras_shape[1:],
                       Dense(output_dim=deepNetSize*deepMaxout, activation=activation, 
                             W_regularizer = l2(l2Reg)), deepMaxout)(layerInput)
        if dropoutH:
            layer = Dropout(dropoutH)(layer)
        layer = Model(layerInput, layer)
        layers += [layer]
    if deepResidual == 'dense':
        mlp = DenseResidual((convNetOutSize,), layers, False, **{'W_regularizer': l2(l2Reg)})
    else:
        mlpInput = Input((convNetOutSize,))
        mlp = mlpInput
        for layer in layers:
            mlp = layer(mlp)
        mlp = Model(mlpInput, mlp)
        if deepResidual:
            mlp = Residual((convNetOutSize,), mlp, True, **{'W_regularizer': l2(l2Reg)})
    model = mlp(model)
    model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                    W_regularizer=l2(l2Reg))(model)
    return model

def makePostureRDANModel(input, numClasses,
                         rnnHiddenSize = 11, rnnOutSize = 11, deepNetSize = 11,
                         numRNNLayers = 1, numDeepLayers = 3,
                         bidirectional=False, l2Reg = 0.0001,
                         activation='relu', useGRU=False,
                         dropoutH = 0,
                         makeSymmetric = False, stripSoftmax = False,
                         poolingMode = 'ave', masking = True,
                         maxout = 1, deepResidual = False):
    """
    Make a recurrent deep averaging network using an LSTM. 
    Might be easier to follow if Keras' functional API is used (done!).

    Maxout and residual effects are not considered for the recurrent layers.
    """
    model = input
    if bidirectional:
        #def fork(mod, n=2):
        #    #obsolete function
        #    forks = []
        #    for i in range(n):
        #        f = Sequential()
        #        f.add(mod)
        #        forks.append(f)
        #    return forks
        rnnSize = rnnOutSize if (numRNNLayers == 1) else rnnHiddenSize
        if makeSymmetric:
            if useGRU:
                forward = GRU(output_dim=rnnSize, return_sequences=True, 
                              W_regularizer=l2(l2Reg))
            else:
                forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                               W_regularizer=l2(l2Reg))
            backward = forward(Reverse()(model))
            forward = forward(model)
        else:
            if useGRU:
                forward = GRU(output_dim=rnnSize, return_sequences=True, 
                                W_regularizer=l2(l2Reg))(model)
                backward = GRU(output_dim=rnnSize, return_sequences=True, 
                                W_regularizer=l2(l2Reg))(Reverse()(model))
            else:
                forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                 W_regularizer=l2(l2Reg))(model)
                backward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                  W_regularizer=l2(l2Reg))(Reverse()(model))
        for i in range(numRNNLayers-1):
            # combine forward and backward LSTMs
            model = merge([forward, Reverse()(backward)], mode = 'concat')
            if dropoutH:
                model = Dropout(dropoutH)(model)
            # split outputs so that the next forward-backward layer can be made
            rnnSize = rnnOutSize if (i >= numRNNLayers-2) else rnnHiddenSize
            if makeSymmetric:
                if useGRU:
                    forward = GRU(output_dim=rnnSize, return_sequences=True, 
                                  W_regularizer=l2(l2Reg))
                else:
                    forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                   W_regularizer=l2(l2Reg))
                backward = forward(Reverse()(model))
                forward = forward(model)
            else:
                if useGRU:
                    forward = GRU(output_dim=rnnSize, return_sequences=True, 
                                    W_regularizer=l2(l2Reg))(model)
                    backward = GRU(output_dim=rnnSize, return_sequences=True, 
                                    W_regularizer=l2(l2Reg))(Reverse()(model))
                else:
                    forward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                     W_regularizer=l2(l2Reg))
                    backward = LSTM(output_dim=rnnSize, return_sequences=True, 
                                      W_regularizer=l2(l2Reg))(Reverse()(model))
        #combine forward and backward for unidirectional feedforward layers
        model = merge([forward, Reverse()(backward)], mode = 'concat')
        if dropoutH:
            model = Dropout(dropoutH)(model)
    else:
        for i in range(numRNNLayers):
            rnnSize = rnnOutSize if (i >= numRNNLayers-1) else rnnHiddenSize
            if useGRU:
                model = GRU(output_dim=rnnSize, return_sequences=True, 
                            W_regularizer=l2(l2Reg))(model)
            else:
                model = LSTM(output_dim=rnnSize, return_sequences=True, 
                             W_regularizer=l2(l2Reg))(model)
            if dropoutH:
                model = Dropout(dropoutH)(model)
    # deep averaging network
    model = MaskEatingLambda(TimeDistributedMerge(isMasked = masking, mode = poolingMode), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(model)
    #for i in range(numDeepLayers-1):
    #    model = Dense(output_dim=deepNetSize, activation=activation, 
    #                  W_regularizer=l2(l2Reg))(model)
    #    if dropoutH:
    #        model = Dropout(dropoutH)(model)
    #model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
    #                W_regularizer=l2(l2Reg))(model)
    #return model
    mlpInSize = rnnOutSize*2 if bidirectional else rnnOutSize
    layers = []
    for i in range(numDeepLayers-1):
        layerInput = Input((mlpInSize if i == 0 else deepNetSize,))
        layer = Maxout(layerInput._keras_shape[1:],
                       Dense(output_dim=deepNetSize*maxout, activation=activation, 
                             W_regularizer = l2(l2Reg)), maxout)(layerInput)
        if dropoutH:
            layer = Dropout(dropoutH)(layer)
        layer = Model(layerInput, layer)
        layers += [layer]
    if deepResidual == 'dense':
        mlp = DenseResidual((mlpInSize,), layers, False, **{'W_regularizer': l2(l2Reg)})
    else:
        mlpInput = Input((mlpInSize,))
        mlp = mlpInput
        for layer in layers:
            mlp = layer(mlp)
        mlp = Model(mlpInput, mlp)
        if deepResidual:
            mlp = Residual((mlpInSize,), mlp, True, **{'W_regularizer': l2(l2Reg)})
    model = mlp(model)
    model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                    W_regularizer=l2(l2Reg))(model)
    return model
    
def makePostureRNNModel(input, numClasses, hiddenSize=11, 
                        numRNNLayers=1, bidirectional=False,
                        l2Reg = 0.0001, useGRU=False,
                        dropoutH = 0,
                        activation='relu', 
                        numDeepLayers=1, deepNetSize=11,
                        makeSymmetric = False, stripSoftmax=False,
                        maxout = 1, deepResidual = False):
    model = input
    if bidirectional:
        returnSequences = numRNNLayers > 1
        if makeSymmetric:
            if useGRU:
                forward = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                              W_regularizer=l2(l2Reg))
            else:
                forward = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                               W_regularizer=l2(l2Reg))
            backward = forward(Reverse()(model))
            forward = forward(model)
        else:
            if useGRU:
                forward = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                              W_regularizer=l2(l2Reg))(model)
                backward = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                               W_regularizer=l2(l2Reg))(Reverse()(model))
            else:
                forward = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                               W_regularizer=l2(l2Reg))(model)
                backward = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                                W_regularizer=l2(l2Reg))(Reverse()(model))
        for i in xrange(numRNNLayers-1):
            # combine forward and backward RNNs
            model = merge([forward, Reverse()(backward)], mode='concat')
            if dropoutH:
                model = Dropout(dropoutH)(model)
            # split outputs so that the next forward-backward layer can be made
            returnSequences = i < numRNNLayers-2
            if makeSymmetric:
                if useGRU:
                    forward = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                                  W_regularizer=l2(l2Reg))
                else:
                    forward = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                                   W_regularizer=l2(l2Reg))
                backward = forward(Reverse()(model))
                forward = forward(model)
            else:
                if useGRU:
                    forward = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                                  W_regularizer=l2(l2Reg))(model)
                    backward = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                                   W_regularizer=l2(l2Reg))(Reverse()(model))
                else:
                    forward = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                                   W_regularizer=l2(l2Reg))(model)
                    backward = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                                    W_regularizer=l2(l2Reg))(Reverse()(model))
        #combine forward and backward for unidirectional feedforward layers
        model = merge([forward, Reverse()(backward)], mode='concat')
        if dropoutH:
            model = Dropout(dropoutH)(model)
    else:
        for i in xrange(numRNNLayers):
            returnSequences = i < numRNNLayers-1 # only return last ouput in final layer
            if useGRU:
                model = GRU(output_dim=hiddenSize, return_sequences=returnSequences, 
                              W_regularizer=l2(l2Reg))(model)
            else:
                model = LSTM(output_dim=hiddenSize, return_sequences=returnSequences, 
                               W_regularizer=l2(l2Reg))(model)  
            if dropoutH:
                model = Dropout(dropoutH)(model)
    # we only care about the last output, but it could show up anywhere
    # since the sequences are of variable length. Therefore, we will
    # rely upon the caller to use an appropriate 'sample_weight' when fitting.
    # OR!!!! We simply set return_sequences to false on the last rnn layer. 
    # This should ignore masked outputs.
    if numDeepLayers > 0:
        #for i in range(numDeepLayers-1):
        #    model = Dense(output_dim=deepNetSize, activation=activation, 
        #                    W_regularizer=l2(l2Reg))(model)
        #    if dropoutH:
        #        model = Dropout(dropoutH)(model)
        #model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
        #              W_regularizer = l2(l2Reg))(model)
        mlpInSize = hiddenSize*2 if bidirectional else hiddenSize
        layers = []
        for i in range(numDeepLayers-1):
            layerInput = Input((mlpInSize if i == 0 else deepNetSize,))
            layer = Maxout(layerInput._keras_shape[1:],
                           Dense(output_dim=deepNetSize*maxout, activation=activation, 
                                 W_regularizer = l2(l2Reg)), maxout)(layerInput)
            if dropoutH:
                layer = Dropout(dropoutH)(layer)
            layer = Model(layerInput, layer)
            layers += [layer]
        if deepResidual == 'dense':
            mlp = DenseResidual((mlpInSize,), layers, False, **{'W_regularizer': l2(l2Reg)})
        else:
            mlpInput = Input((mlpInSize,))
            mlp = mlpInput
            for layer in layers:
                mlp = layer(mlp)
            mlp = Model(mlpInput, mlp)
            if deepResidual:
                mlp = Residual((mlpInSize,), mlp, True, **{'W_regularizer': l2(l2Reg)})
        model = mlp(model)
        model = Dense(output_dim=numClasses, activation='softmax' if not stripSoftmax else activation, 
                        W_regularizer=l2(l2Reg))(model)
    return model

def makeAutoencoder(input, maxNumTimesteps, numFeatures, 
                    encodedSize=11, hiddenSize=11, 
                    numRNNLayers=1, bidirectional=False,
                    l2Reg = 0.0001, 
                    dropoutH = 0,
                    activation='relu', 
                    numDeepLayers=1, deepNetSize=11,
                    makeSymmetric = False):
    encoder = makePostureRNNModel(input, numClasses=encodedSize,
                                  hiddenSize=hiddenSize, 
                                  numRNNLayers=numRNNLayers, bidirectional=bidirectional,
                                  l2Reg = l2Reg, useGRU=True,
                                  dropoutH = dropoutH, activation=activation, 
                                  numDeepLayers=numDeepLayers, deepNetSize=deepNetSize,
                                  makeSymmetric = makeSymmetric, stripSoftmax = True)
    autoencoder = RepeatVector(maxNumTimesteps)(encoder)
    #for j in xrange(numDeepLayers):
    #    decoder = TimeDistributed(Dense(deepNetSize, activation = activation, W_regularizer = l2(l2Reg)))(autoencoder)
    for i in xrange(numRNNLayers):
        autoencoder = GRU(hiddenSize, return_sequences = True, 
                          W_regularizer= l2(l2Reg))(autoencoder)
        if dropoutH and i < numRNNLayers - 1:
            autoencoder = Dropout(dropoutH)(autoencoder)
    autoencoder = TimeDistributed(Dense(numFeatures, activation = 'linear'))(autoencoder)
    return encoder, autoencoder


def augmentPostureDataset(sequences, classes, dataRange, amount):
    """
    Randomly permutes sequences in the provided data range and adds them to the
    dataset, thereby simulating additional data.
    """
    def permutePosture(posture, length):
        perm = np.random.permutation(length).tolist()
        perm = perm + range(length, posture.shape[0])
        return posture[perm,:]
    #choose which postures will be randomly permuted by index
    selection = np.random.choice(dataRange, amount)
    #get a permuted copy of each posture
    pps = []
    classLists = [] #if efficiency is an issue, could be removed entirely
    for s in selection:
        classList = classes[:, s]
        length = (classList >= 0).sum()
        pp = permutePosture(sequences[:, s, :], length)
        pps += [pp]
        classLists += [classList]
    #augment data
    dataRange += [r + sequences.shape[1] for r in range(amount)]
    classes = np.concatenate((classes, np.array(classLists).T), axis = 1) 
    pps = np.stack(pps).transpose((1,0,2))
    sequences = np.concatenate((sequences, pps), axis = 1)
    return sequences, classes, dataRange

def comprehensivePostEvaluation(directory=defaultDirectory(), trainPer=0.6, 
                                valPer=0.25, testPer=0.15, totalPer=1, 
                                randSeed = 1, modelType = 'cdan', 
                                numSpecialLayers = 2, numDeepLayers = 3,
                                numSpecialNodes = 11, numDeepNodes = 11,
                                numSpecialOut = 11, useGRU=False,
                                l2Reg = 0.0001, activation='relu',
                                dropoutI=0, dropoutH=0,
                                batchSize=64, numEpochs = 500, 
                                learningRate = 0.001, trainMode = 'continue',
                                modelFile = None, 
                                trainAbs = None, valAbs = None, testAbs = None,
                                augmentData = None, wordDropout=0.2,
                                gaussianNoise = 20, makeSymmetric=False,
                                poolingMode = 'ave', masking = True,
                                center = True, prependMean = False): 
    
    np.random.seed(randSeed)
    struct = loadDataset(directory=directory, delRange = None, trainPer = trainPer,
                         valPer=valPer, testPer=testPer, totalPer=totalPer, classAbbrv='c',
                         prune = False, classRange=range(1,6), preExt='.unw',
                         trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs)
    struct = list(struct)
    """
    For non-convolutional networks, the order of the markers matters. 
    Therefore, simulate additional data by permuting markers.
    """
    if modelType != 'cdan' and augmentData is not None:
        struct[0], struct[1], struct[2] = augmentPostureDataset(struct[0], struct[1],
                                                                struct[2], augmentData)
        struct[7] += augmentData
    """
    Sort markers by position.
    """
    struct[0] = sortData(struct[0].transpose([1,0,2]), np.eye(3), True).transpose([1,0,2])
    """
    Center each frame on its mean.
    """
    if center:
        struct[0] = centerData(struct[0].transpose([1,0,2]), True, prependMean).transpose([1,0,2])
    """
     CDAN only outputs a single class per sequence rather than an output for
     each timestep of the sequence. Fix the target class ID matrix to account
     for this.
     Could something similar be done for the RNNs? Yes!
    """
    struct[1] = struct[1][0,:].reshape((1, struct[7]))
    
    inputLayer = Input(shape=(struct[6] + (1 if prependMean else 0), 3))
    model = inputLayer
    if wordDropout:
        model = WordDropout(wordDropout, False)(model)
    if masking:
        model = Masking(0)(model)
    if gaussianNoise:
        model = addRealGaussianNoise(model, gaussianNoise, masking)
    if dropoutI:
        model = Dropout(dropoutI)(model)

    model = buildPostureModel(input = model, modelType=modelType, numSpecialLayers=numSpecialLayers,
                              numDeepLayers=numDeepLayers, numSpecialNodes=numSpecialNodes,
                              numDeepNodes=numDeepNodes, numSpecialOut=numSpecialOut,
                              useGRU=useGRU, l2Reg=l2Reg, 
                              dropoutH=dropoutH, activation=activation,
                              numClasses=struct[5],
                              makeSymmetric = makeSymmetric, poolingMode = poolingMode)
    model = Model(input=inputLayer, output=model)
    if modelFile is None:
        modelFile = namePostureModel(randSeed,modelType=modelType, numSpecialLayers=numSpecialLayers,
                                     numDeepLayers=numDeepLayers, numSpecialNodes=numSpecialNodes,
                                     numDeepNodes=numDeepNodes, numSpecialOut=numSpecialOut,
                                     useGRU=useGRU, l2Reg=l2Reg, dropoutI=dropoutI,
                                     dropoutH=dropoutH, activation=activation,
                                     augmentData=augmentData,
                                     makeSymmetric = makeSymmetric, poolingMode = poolingMode,
                                     masking = masking, center = center, prependMean = prependMean)
    #sampleWeights = None if (modelType not in ['rnn', 'birnn']) else weightSamplesByTimeDecay(y_true=struct[1]).T
    custom_objects = {'MaskEatingLambda': MaskEatingLambda, 'WordDropout': WordDropout, 'Reverse': Reverse, 'MaskedPooling': MaskedPooling, 'Sort': Sort} 
    trainKerasModel(model=model, batchSize=batchSize,
                    numEpochs=numEpochs, learningRate=learningRate,
                    sequences=[struct[0]], classes = struct[1], trainRange=struct[2],
                    valRange = struct[3], testRange = struct[4],
                    numClasses = struct[5], 
                    modelFile = modelFile, callbacks = [EarlyStopping(patience=20)],
                    sampleWeights=None, 
                    outDirectory='', trainMode=trainMode,
                    custom_objects = custom_objects, 
                    optimizer = RMSProp(learningRate))

def comprehensivePostLOOEvaluation(directory=defaultDirectory(), trainPer=0.6, 
                                   valPer=0.25, testPer=0.15, totalPer=1, 
                                   randSeed = 1, modelType = 'cdan',
                                   numSpecialLayers = 2, numDeepLayers = 3,
                                   numSpecialNodes = 11, numDeepNodes = 11,
                                   numSpecialOut = 11, useGRU=False,
                                   l2Reg = 0.0001, activation='relu',
                                   dropoutI = 0, dropoutH = 0,
                                   modelFilePrefix = '', trainMode = 'continue',
                                   randSeed2 = None, numEpochs = 500,
                                   learningRate = 0.001, batchSize=64,
                                   trainAbs = 75, valAbs = 75, testAbs = 75,
                                   augmentData = None, wordDropout=0.2,
                                   gaussianNoise = 20, makeSymmetric=False,
                                   poolingMode = 'ave', masking = True,
                                   center = True, prependMean = False,
                                   maxout = 1, embeddingResidual = False, deepResidual = False,
                                   patience = 20, saveResults = False, justResults = False): 
    
    u=0
    outDirectory = namePostureModel(randSeed, modelType=modelType, numSpecialLayers=numSpecialLayers,
                                    numDeepLayers=numDeepLayers, numSpecialNodes=numSpecialNodes,
                                    numDeepNodes=numDeepNodes, numSpecialOut=numSpecialOut,
                                    useGRU=useGRU, l2Reg=l2Reg, dropoutI=dropoutI,
                                    dropoutH=dropoutH, activation=activation,
                                    trainPer=trainPer, valPer=valPer, testPer=testPer,
                                    trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs,
                                    augmentData=augmentData,
                                    wordDropout = wordDropout, gaussianNoise = gaussianNoise,
                                    makeSymmetric = makeSymmetric, poolingMode = poolingMode,
                                    masking = masking, center = center, prependMean = prependMean,
                                    maxout = maxout, embeddingResidual = embeddingResidual, deepResidual = deepResidual)
    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)
    resultsFile = outDirectory + '\\' + modelFilePrefix + 'Results.pkl'
    if trainMode == 'continue' and os.path.isfile(resultsFile):
        import dill
        #load previous results to which we will append results for new users
        with open(resultsFile, 'rb') as f:
            data = dill.load(f)
            losses = data['losses']
            accs = data['acc']
            histories = data['histories']
            try:
                users = data['users']
                if 12 in users:
                    losses = np.asarray(losses)[0:12]
                    accs = np.asarray(accs)[0:12]
                    histories = [history for history, i in zip(histories, range(len(histories))) if i < 12]
                users = range(len(losses))
            except KeyError:
                users = range(len(losses))
            # restore losses and accs as lists instead of numpy arrays if saved as numpy arrays
            losses = [loss for loss in losses]
            # restore unscaled accuracies if originally saved as scaled
            accs = np.asarray(accs)/100 if np.max(accs) > 1 else accs
            accs = [acc for acc in accs]
    else:
        losses = []
        accs = []
        histories = []
        users = []
    if (isinstance(poolingMode, list) or isinstance(poolingMode, tuple)) and modelType != 'pmeq':
        poolingMode = poolingMode[0]
    try:
        maxout = list(maxout)
        if not (modelType == 'cdan' or modelType == 'pmeq' or modelType == 'fpmeq'):
            #maxout being a list doesn't make sense if not a CDAN or PMEQ
            maxout = maxout[0]
    except:
        #maxout isn't a list. This is okay.
        pass
    if modelType == 'cdan' and numSpecialLayers == 0:
        numSpecialOut = 3

    if justResults:
        return histories, np.asarray(losses), np.asarray(accs)*100

    np.random.seed(randSeed)
    structs = loadDataset(directory=directory, delRange = None, trainPer = trainPer,
                         valPer=valPer, testPer=testPer, totalPer=totalPer, classAbbrv='c',
                         prune = False, classRange=range(1,6), preExt='.unw',
                         LOUO = True,
                         trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs)
    
    if randSeed2 is not None: #control randomization of training 
        np.random.seed(randSeed2)
    for struct in structs:
        modelFile = modelFilePrefix + 'LOU-' + str(u) +'-'
        modelFile = modelFile
        if (os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'.json') 
            and os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'_Weights.h5')):
            #if we have already trained for leaving out this user
            if trainMode == 'continue': #continue until each user has a model
                trainMode2 = 'skip' 
            elif trainMode == 'continue-each': # continue training previous models
                trainMode2 = 'continue'
            else:
                trainMode2 = 'overwrite'
        else:
            trainMode2 = trainMode
            
        if trainMode2 != 'skip' or not os.path.isfile(resultsFile) or u not in users:
            struct = list(struct)
            """
            For non-convolutional networks, the order of the markers matters. 
            Therefore, simulate additional data by permuting markers.
            """
            if modelType != 'cdan' and augmentData is not None:
                struct[0], struct[1], struct[2] = augmentPostureDataset(struct[0], struct[1],
                                                                        struct[2], augmentData)
                struct[7] += augmentData
            """
            Sort markers by position.
            """
            struct[0] = sortData(struct[0].transpose([1,0,2]), np.eye(3), True).transpose([1,0,2])
            """
            Center each frame on its mean.
            """
            if center:
                struct[0] = centerData(struct[0].transpose([1,0,2]), True, prependMean).transpose([1,0,2])
            """
             DANs only outputs a single class per sequence rather than an output for
             each timestep of the sequence. Fix the target class ID matrix to account
             for this.
             Could something similar be done for the RNNs? Yes!
            """
            struct[1] = struct[1][0,:].reshape((1, struct[7]))
        
            inputLayer = Input(shape=(struct[6] + (1 if prependMean else 0), 3))
            model = inputLayer
            if wordDropout:
                model = WordDropout(wordDropout, False)(model)
            if masking:
                model = Masking(0)(model)
            if gaussianNoise:
                model = addRealGaussianNoise(model, gaussianNoise, masking)
            if dropoutI:
                model = Dropout(dropoutI)(model)

            kwargs = {} if modelType != 'fpmeq' else {'batchSize': batchSize}
            model = buildPostureModel(input = model, modelType=modelType, numSpecialLayers=numSpecialLayers,
                                      numDeepLayers=numDeepLayers, numSpecialNodes=numSpecialNodes,
                                      numDeepNodes=numDeepNodes, numSpecialOut=numSpecialOut,
                                      useGRU=useGRU, l2Reg=l2Reg, 
                                      dropoutH=dropoutH, activation=activation,
                                      numClasses=struct[5],
                                      makeSymmetric = makeSymmetric, poolingMode = poolingMode,
                                      masking = masking, maxout = maxout, 
                                      embeddingResidual = embeddingResidual, deepResidual = deepResidual,
                                      **kwargs)
            model = Model(input = inputLayer, output = model)
            if modelFile is None:
                modelFile = namePostureModel(randSeed,modelType=modelType, numSpecialLayers=numSpecialLayers,
                                             numDeepLayers=numDeepLayers, numSpecialNodes=numSpecialNodes,
                                             numDeepNodes=numDeepNodes, numSpecialOut=numSpecialOut,
                                             useGRU=useGRU, l2Reg=l2Reg, dropoutI=dropoutI,
                                             dropoutH=dropoutH, activation=activation,
                                             trainPer=trainPer, valPer=valPer, testPer=testPer,
                                             trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs,
                                             augmentData=augmentData, 
                                             wordDropout = wordDropout, gaussianNoise = gaussianNoise,
                                             makeSymmetric = makeSymmetric, 
                                             masking = masking, center = center, prependMean = prependMean,
                                             maxout = maxout, embeddingResidual = embeddingResidual, deepResidual = deepResidual)
            #sampleWeights = None if (modelType not in ['rnn', 'birnn']) else weightSamplesByTimeDecay(y_true=struct[1].T)
            custom_objects = getCustomObjects()
            history, loss, acc = trainKerasModel(model=model, batchSize=batchSize,
                                                 numEpochs=numEpochs, 
                                                 sequences=[struct[0]], classes = struct[1], trainRange=struct[2],
                                                 valRange = struct[3], testRange = struct[4],
                                                 numClasses = struct[5], 
                                                 modelFile = modelFile, callbacks = [EarlyStopping(patience=patience)],
                                                 sampleWeights=None, 
                                                 outDirectory=outDirectory, trainMode=trainMode2,
                                                 custom_objects = custom_objects, 
                                                 optimizer = RMSprop(learningRate))
            #catch our breath.... Really, give the user a chance to insert Ctrl-C
            time.sleep(2)
            losses += [loss]
            accs += [acc]
            histories += [history]
            users += [u]
            if saveResults:
                import dill
                with open(resultsFile, 'wb') as f:
                    data = {'losses': losses, 'acc': accs, 'histories': histories, 'users': users}
                    dill.dump(data, f)
        u += 1
    if saveResults:
        import dill
        with open(resultsFile, 'wb') as f:
            data = {'losses': losses, 'acc': accs, 'histories': histories, 'users': users}
            dill.dump(data, f)
    losses = np.asarray(losses)
    accs = np.asarray(accs)*100
    trainPer, valPer, _, _ = normalizePercentages(trainPer, valPer, 0, 1)
    sys.stdout.write('\n')
    sys.stdout.write('**********\n')
    sys.stdout.write('Leave One User Out Evaluation, ' + modelAlias(modelType, makeSymmetric) + '\n' 
                     +'Test Results for ' + outDirectory + '\n'
                     + str(numEpochs) + ' Maximum Epochs at ' + ("%0.2f" % trainPer) + '/' + ("%0.2f" % valPer) + ' Training/Validation Split\n')
    sys.stdout.write('\n')
    sys.stdout.write('Loss: ' + str(np.mean(losses)) + ' +/- ' + str(np.std(losses)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Loss: ' + str(np.percentile(losses, 25))
                     + ', ' + str(np.median(losses)) 
                     +  ', ' + str(np.percentile(losses, 75)) +'\n')
    sys.stdout.write('\n')
    sys.stdout.write('Accuracy: ' + str(np.mean(accs)) + ' +/- ' + str(np.std(accs)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Accuracy: ' + str(np.percentile(accs, 25))
                     + ', ' + str(np.median(accs)) 
                     +  ', ' + str(np.percentile(accs, 75)) +'\n')
    return histories, losses, accs

def buildPostureModel(numClasses, input, modelType = 'cdan',
                      numSpecialLayers = 2, numDeepLayers = 3,
                      numSpecialNodes = 11, numDeepNodes = 11,
                      numSpecialOut = 11, useGRU=False,
                      l2Reg = 0.0001, dropoutH = 0,
                      activation='relu', makeSymmetric = False,
                      stripSoftmax = False, poolingMode = 'ave',
                      masking = True, maxout = 1,
                      embeddingResidual = False, deepResidual = False, 
                      **kwargs):
    
    modelType = modelType.lower()
    modelChoices = ['dcnn', 'rcnn', 'bircnn', 'cdan', 'rdan', 'birdan', 'rnn', 'birnn', 'pmeq','fpmeq']   
    if modelType in modelChoices:
        def getModel(c):
            return { 
                'dcnn': lambda: makePostureDeepCNNModel(input = input,
                                                        numConvLayers=numSpecialLayers,
                                                        numFilters=numSpecialNodes,
                                                        numClasses=numClasses,
                                                        l2Reg=l2Reg,
                                                        activation=activation,
                                                        dropoutH = dropoutH, 
                                                        stripSoftmax=stripSoftmax,
                                                        poolingMode = poolingMode,
                                                        poolSize=numSpecialOut,
                                                        numDeepLayers = numDeepLayers,
                                                        deepNetSize = numDeepNodes,
                                                        masking = masking, 
                                                        **kwargs),
                'rcnn': lambda: makePostureRCNNModel(input = input,
                                                     numRNNLayers=numSpecialLayers,
                                                     rnnHiddenSize=numSpecialNodes,
                                                     numClasses=numClasses,
                                                     l2Reg=l2Reg,
                                                     activation=activation,
                                                     dropoutH = dropoutH, 
                                                     stripSoftmax=stripSoftmax,
                                                     poolingMode = poolingMode,
                                                     poolSize=numSpecialOut,
                                                     numDeepLayers = numDeepLayers,
                                                     deepNetSize = numDeepNodes,
                                                     useGRU = useGRU,
                                                     bidirectional = False,
                                                     masking = masking,
                                                     **kwargs),
                'bircnn': lambda: makePostureRCNNModel(input = input,
                                                       numRNNLayers=numSpecialLayers,
                                                       rnnHiddenSize=numSpecialNodes,
                                                       numClasses=numClasses,
                                                       l2Reg=l2Reg,
                                                       activation=activation,
                                                       dropoutH = dropoutH, 
                                                       stripSoftmax=stripSoftmax,
                                                       poolingMode = poolingMode,
                                                       poolSize=numSpecialOut,
                                                       numDeepLayers = numDeepLayers,
                                                       deepNetSize = numDeepNodes,
                                                       useGRU = useGRU,
                                                       bidirectional = True,
                                                       masking = masking,
                                                       **kwargs),
                'cdan': lambda: makePostureCDANModel(input = input,
                                                     numCNNLayers=numSpecialLayers,
                                                     numDeepLayers=numDeepLayers,
                                                     convNetHiddenSize=numSpecialNodes,
                                                     convNetOutSize=numSpecialOut,
                                                     deepNetSize=numDeepNodes,
                                                     numClasses=numClasses,
                                                     l2Reg=l2Reg,
                                                     activation=activation,
                                                     dropoutH = dropoutH, 
                                                     stripSoftmax=stripSoftmax,
                                                     poolingMode = poolingMode,
                                                     masking = masking,
                                                     maxout = maxout,
                                                     convNetResidual = embeddingResidual,
                                                     deepResidual = deepResidual,
                                                     **kwargs),
                'fpmeq': lambda: makeFPMEQModel(input = input,
                                              fNumLayers=numSpecialLayers,
                                              numDeepLayers=numDeepLayers,
                                              fHiddenSize=numSpecialNodes,
                                              fOutSize=numSpecialOut,
                                              deepNetSize=numDeepNodes,
                                              numClasses=numClasses,
                                              l2Reg=l2Reg,
                                              activation=activation,
                                              dropoutH = dropoutH, 
                                              stripSoftmax=stripSoftmax,
                                              poolingMode = poolingMode,
                                              masking = masking,
                                              maxout = maxout,
                                              deepResidual = deepResidual,
                                              **kwargs),
                'pmeq': lambda: makePMEQModel(input = input,
                                              fNumLayers=numSpecialLayers,
                                              numDeepLayers=numDeepLayers,
                                              fHiddenSize=numSpecialNodes,
                                              fOutSize=numSpecialOut,
                                              deepNetSize=numDeepNodes,
                                              numClasses=numClasses,
                                              l2Reg=l2Reg,
                                              activation=activation,
                                              dropoutH = dropoutH, 
                                              stripSoftmax=stripSoftmax,
                                              poolingMode = poolingMode,
                                              masking = masking,
                                              maxout = maxout,
                                              fResidual = embeddingResidual,
                                              deepResidual = deepResidual,
                                              **kwargs),
                'rdan': lambda: makePostureRDANModel(input = input,
                                                     numRNNLayers=numSpecialLayers,
                                                     numDeepLayers=numDeepLayers,
                                                     rnnHiddenSize=numSpecialNodes,
                                                     rnnOutSize=numSpecialOut,
                                                     deepNetSize=numDeepNodes,
                                                     useGRU=useGRU,
                                                     numClasses=numClasses,
                                                     l2Reg=l2Reg,
                                                     activation=activation,
                                                     dropoutH = dropoutH, 
                                                     stripSoftmax=stripSoftmax,
                                                     poolingMode = poolingMode,
                                                     masking = masking,
                                                     maxout = maxout,
                                                     deepResidual = deepResidual,
                                                     **kwargs),
                'birdan': lambda: makePostureRDANModel(input = input,
                                                       numRNNLayers=numSpecialLayers,
                                                       numDeepLayers=numDeepLayers,
                                                       rnnHiddenSize=numSpecialNodes,
                                                       rnnOutSize=numSpecialOut,
                                                       deepNetSize=numDeepNodes,
                                                       bidirectional=True,
                                                       useGRU=useGRU,
                                                       numClasses=numClasses,
                                                       l2Reg=l2Reg,
                                                       activation=activation,
                                                       dropoutH = dropoutH,
                                                       makeSymmetric = makeSymmetric, 
                                                       stripSoftmax=stripSoftmax,
                                                       poolingMode = poolingMode,
                                                       masking = masking,
                                                       maxout = maxout,
                                                       deepResidual = deepResidual,
                                                       **kwargs),
                'rnn': lambda: makePostureRNNModel(input = input,
                                                   numRNNLayers=numSpecialLayers,
                                                   numDeepLayers=numDeepLayers,
                                                   deepNetSize = numDeepNodes,
                                                   hiddenSize = numSpecialNodes,
                                                   useGRU=useGRU,
                                                   numClasses=numClasses,
                                                   l2Reg=l2Reg,
                                                   dropoutH = dropoutH, 
                                                   stripSoftmax=stripSoftmax,
                                                   maxout = maxout,
                                                   deepResidual = deepResidual,
                                                   **kwargs),
                'birnn': lambda: makePostureRNNModel(input = input,
                                                     numRNNLayers=numSpecialLayers,
                                                     numDeepLayers=numDeepLayers,
                                                     deepNetSize = numDeepNodes,
                                                     hiddenSize = numSpecialNodes,
                                                     bidirectional=True,
                                                     useGRU=useGRU,
                                                     numClasses=numClasses,
                                                     l2Reg=l2Reg,
                                                     dropoutH = dropoutH,
                                                     makeSymmetric = makeSymmetric, 
                                                     stripSoftmax=stripSoftmax,
                                                     maxout = maxout,
                                                     deepResidual = deepResidual,
                                                     **kwargs)
            }[c]   
        
        model = getModel(modelType)()
        return model
    else:
        sys.stderr.write('Unknown model type. Please choose one of the following:')
        for choice in modelChoices:
            sys.stderr.write(choice)
        raise ValueError('Unknown model type.')


def namePostureModel(randSeed, trainPer=0.6, valPer=0.25, 
                     testPer=0.15, totalPer=1, modelType = 'cdan',
                     numSpecialLayers = 2, numDeepLayers = 3,
                     numSpecialNodes = 11, numDeepNodes = 11,
                     numSpecialOut = 11, useGRU=False,
                     l2Reg = 0.0001, dropoutI = 0, dropoutH = 0,
                     activation='relu', 
                     trainAbs = None, valAbs = None, testAbs = None,
                     augmentData=None, 
                     gaussianNoise = 20, wordDropout = 0.2,
                     makeSymmetric = False, poolingMode = 'ave',
                     masking = True, center = True, prependMean = False,
                     maxout = 1, embeddingResidual = False, deepResidual = False):
                         
    modelType = modelType.lower()
    modelChoices = ['dcnn', 'rcnn', 'bircnn', 'cdan', 'rdan', 'birdan', 'rnn', 'birnn', 'pmeq', 'fpmeq']
    poolingMode = str(poolingMode)
    if poolingMode == 'ave':
        poolingMode = '' #default poolingMode. Ignore it in name
    if modelType in modelChoices:
        def getModelFile(c):
            return { 
                'dcnn': "-".join(['DCNN', 
                                  'L', str(numSpecialLayers), str(numSpecialOut), str(numDeepLayers),
                                  'N', str(numSpecialNodes), str(numDeepNodes),
                                  'A', activation]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else ''),
                'rcnn': "-".join(['RCNN', ('GRU' if useGRU else 'LSTM'),
                                  'L', str(numSpecialLayers), str(numSpecialOut), str(numDeepLayers),
                                  'N', str(numSpecialNodes), str(numDeepNodes)]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else ''),
                'bircnn': "-".join(['BiRCNN' if not makeSymmetric else 'SBiRCNN',
                                    ('GRU' if useGRU else 'LSTM'), 
                                    'L', str(numSpecialLayers), str(numSpecialOut), str(numDeepLayers),
                                    'N', str(numSpecialNodes), str(numDeepNodes)]) 
                                    + (('-' + poolingMode + '-') if len(poolingMode) > 0 else ''),
                'cdan': "-".join(['CDAN', 
                                  'L', str(numSpecialLayers), str(numDeepLayers), 
                                  'N', 
                                  (((str(numSpecialNodes)+'-'+str(numSpecialOut)) 
                                   if numSpecialLayers > 1 else (str(numSpecialOut) if numSpecialLayers > 0 else ''))
                                   + '-' + str(numDeepNodes) 
                                   if numDeepLayers > 1 else '')]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else '')
                                  + (('-R' if embeddingResidual or deepResidual else '') 
                                     + ('dE' if embeddingResidual == 'dense' else ('E' if embeddingResidual else ''))
                                     + ('dD' if deepResidual == 'dense' else ('D' if deepResidual else ''))),
                'fpmeq': "-".join(['FPMEQ', 
                                  'L', str(numSpecialLayers), str(numDeepLayers), 
                                  'N', 
                                  (((str(numSpecialNodes)+'-'+str(numSpecialOut)) 
                                   if numSpecialLayers > 1 else (str(numSpecialOut) if numSpecialLayers > 0 else ''))
                                   + '-' + str(numDeepNodes) 
                                   if numDeepLayers > 1 else '')]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else '')
                                  + ('-RdD' if deepResidual == 'dense' else ('-RD' if deepResidual else '')),
                'pmeq': "-".join(['PMEQ', 
                                  'L', str(numSpecialLayers), str(numDeepLayers), 
                                  'N', 
                                  (((str(numSpecialNodes)+'-'+str(numSpecialOut)) 
                                   if numSpecialLayers > 1 else (str(numSpecialOut) if numSpecialLayers > 0 else ''))
                                   + '-' + str(numDeepNodes) 
                                   if numDeepLayers > 1 else '')]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else '')
                                  + (('-R' if embeddingResidual or deepResidual else '') 
                                     + ('dE' if embeddingResidual == 'dense' else ('E' if embeddingResidual else ''))
                                     + ('dD' if deepResidual == 'dense' else ('D' if deepResidual else ''))),
                'rdan': "-".join(['RDAN', ('GRU' if useGRU else 'LSTM'), 
                                  'L', str(numSpecialLayers), str(numDeepLayers), 
                                  'N', 
                                  (((str(numSpecialNodes)+'-'+str(numSpecialOut)) 
                                   if numSpecialLayers > 1 else str(numSpecialOut))
                                   + '-' + str(numDeepNodes) 
                                   if numDeepLayers > 1 else '')]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else '')
                                  + ('-RdD' if deepResidual == 'dense' else ('-RD' if deepResidual else '')),
                'birdan': "-".join([('BiRDAN' if not makeSymmetric else 'SBiRDAN'), ('GRU' if useGRU else 'LSTM'), 
                                  'L', str(numSpecialLayers), str(numDeepLayers), 
                                  'N', 
                                  (((str(numSpecialNodes)+'-'+str(numSpecialOut)) 
                                   if numSpecialLayers > 1 else str(numSpecialOut))
                                   + '-' + str(numDeepNodes) 
                                   if numDeepLayers > 1 else '')]) 
                                  + (('-' + poolingMode + '-') if len(poolingMode) > 0 else '')
                                  + ('-RdD' if deepResidual == 'dense' else ('-RD' if deepResidual else '')),
                'rnn': "-".join(['RNN', ('GRU' if useGRU else 'LSTM'), 
                                  'L', str(numSpecialLayers), str(numDeepLayers),
                                  'N', 
                                  ((str(numSpecialNodes) + '-' + str(numDeepNodes)) 
                                   if numDeepLayers > 1 else str(numSpecialNodes))])
                                  + ('-RdD' if deepResidual == 'dense' else ('-RD' if deepResidual else '')),
                'birnn': "-".join([('BiRNN' if not makeSymmetric else 'SBiRNN'), ('GRU' if useGRU else 'LSTM'), 
                                  'L', str(numSpecialLayers), str(numDeepLayers),
                                  'N', 
                                  ((str(numSpecialNodes) + '-' + str(numDeepNodes)) 
                                   if numDeepLayers > 1 else str(numSpecialNodes))])
                                  + ('-RdD' if deepResidual == 'dense' else ('-RD' if deepResidual else '')),
            }[c] 
        [trainPer, valPer, 
         testPer, totalPer] = normalizePercentages(trainPer, valPer, 
                                                   testPer, totalPer)
        modelFile = (getModelFile(modelType)
                    + '-A-' + activation #+ ('-' + deepActivation if (modelType == 'cdan' and deepActivation is not None) else '')
                    + (('-L2-' + str(l2Reg)) if l2Reg else '')
                    + (('-D-' + str(dropoutI) +'-'+ str(dropoutH)) 
                        if (dropoutI or dropoutH) else '')
                    + (('-WD-'+str(wordDropout)) if wordDropout else '')
                    + (('-G-' + str(gaussianNoise)) if gaussianNoise else '')
                    + ('-M' if masking else '')
                    + (('-C' + ('P' if prependMean else '')) if center else '')
                    + ('-mxo' + str(maxout) if maxout > 1 else ''))
        modelFile += '-S-' + str(randSeed)
        flag = False
        if trainAbs is not None and valAbs is not None:
            modelFile = '-'.join([modelFile, 'TS', str(trainAbs), str(valAbs)])
        elif (trainAbs is not None) ^ (valAbs is not None):
            raise ValueError('Both trainAbs and valAbs must be set together.')
        else:
            modelFile = '-'.join([modelFile, 'TS', str(trainPer), str(valPer)])
            flag = True
        if testAbs is not None:
            modelFile += '-'+str(testAbs)+ (('-'+str(totalPer)) if flag else '')
        else:
            modelFile += '-'+str(testPer) + '-' + str(totalPer)
        if augmentData is not None:
            modelFile += '-AD-' +  str(augmentData)
        return modelFile
    else:
        sys.stderr.write('Unknown model type. Please choose one of the following:')
        for choice in modelChoices:
            sys.stderr.write(choice)
        raise ValueError('Unknown model type.')
    
def modelAlias(modelType, makeSymmetric = False):
    modelType = modelType.lower()
    modelChoices = ['dcnn', 'rcnn', 'bircnn', 'cdan', 'rdan', 'birdan', 'rnn', 'birnn', 'pmeq', 'fpmeq']   
    if modelType in modelChoices:
        def getAlias(c):
            return { 
                'dcnn': 'DeepCNN',
                'rcnn': 'RCNN',
                'bircnn': 'BiRCNN' if not makeSymmetric else 'SBiRCNN',
                'cdan': 'CDAN',
                'rdan': 'RDAN',
                'birdan': 'BiRDAN' if not makeSymmetric else 'SBiRDAN',
                'rnn': 'RNN',
                'birnn': 'BiRNN' if not makeSymmetric else 'SBiRNN',
                'pmeq': 'PMEQ',
                'fpmeq': 'FPMEQ',
            }[c]   
        
        return getAlias(modelType)
    else:
        return modelType.upper()


def cnnTest(directory = defaultDirectory(),trainPer=0.6, 
            valPer=0.25, testPer=0.15, totalPer=1, 
            randSeed = 1, trainMode = 'continue',
            randSeed2 = None, numEpochs = 60,
            learningRate = 0.1, batchSize=32,
            trainAbs = 75, valAbs = 75, testAbs = 75,
            modelFilePrefix = '', prependMean = False,
            justMLP = False):
    """
    Reproduce Ademola's work for verification and comparison.
    """
    from keras.layers import Conv1D, Conv2D, Reshape, Permute, Activation, Flatten
    np.random.seed(randSeed)
    structs = loadDataset(directory=directory, delRange = None, trainPer = trainPer,
                         valPer=valPer, testPer=testPer, totalPer=totalPer, classAbbrv='c',
                         prune = False, classRange=range(1,6), preExt='.unw',
                         LOUO = True,
                         trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs)
    u=0
    losses = []
    accs = []
    cmEpochs = []
    
    outDirectory = 'AdemolaCNN'
    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)
    if randSeed2 is not None: #control randomization of training 
        np.random.seed(randSeed2)
    for struct in structs:
        modelFile = modelFilePrefix + 'LOU-' + str(u) +'-'
        modelFile = modelFile + outDirectory
        u += 1
        if (os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'.json') 
            and os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'_Weights.h5')):
            #if we have already trained for leaving out this user
            if trainMode == 'continue': #continue until each user has a model
                trainMode2 = 'skip' 
            elif trainMode == 'continue-each': # continue training previous models
                trainMode2 = 'continue'
            else:
                trainMode2 = 'overwrite'
        else:
            trainMode2 = trainMode
        struct = list(struct)
        """
        Sort markers by position.
        """
        struct[0] = sortData(struct[0].transpose([1,0,2]), np.eye(3), True).transpose([1,0,2])
        """
        Center all frames on their mean.
        """
        struct[0] = centerData(struct[0].transpose([1,0,2]), True, prependMean).transpose([1,0,2])
        """
        Normalize markers by position.
        """
        #struct[0] = normalizeData(struct[0].transpose([1,0,2]), True).transpose([1,0,2])
        """
         DANs only outputs a single class per sequence rather than an output for
         each timestep of the sequence. Fix the target class ID matrix to account
         for this.
         Could something similar be done for the RNNs? Yes!
        """
        struct[1] = struct[1][0,:].reshape((1, struct[7]))
        
        maxNumMarkers = struct[6] + (1 if prependMean else 0)
        numClasses = struct[5]
        inputLayer = Input(shape=(maxNumMarkers,3))
        model = inputLayer

        gaussianNoise = False
        if gaussianNoise:
            addRealGaussianNoise(model, 20, False)

        flat = False
        L2Reg = 0.01
        if not justMLP:
            if flat:
                model = Reshape((1, 1, maxNumMarkers*3))(model)
            else:
                model = Reshape((maxNumMarkers, 1, 3))(model)
                model = Permute((3,2,1))(model)
            model = Conv2D(32, 1, 1, activation = 'relu', W_regularizer = l2(L2Reg))(model)
            model = Conv2D(32, 1, 3, activation = 'relu', W_regularizer = l2(L2Reg))(model)
        else:
            model = Flatten()(model)
            model = Dense(36, activation = 'relu', W_regularizer = l2(L2Reg))(model)
        model = Dropout(0.1)(model)
        if not justMLP:
            model = Flatten()(model)
        model = Dense(128, activation = 'relu', W_regularizer = l2(L2Reg))(model)
        model = Dropout(0.1)(model)
        model = Dense(numClasses, activation = 'softmax', W_regularizer = l2(L2Reg))(model)

        model = Model(input = inputLayer, output = model)
        if modelFile is None:
            modelFile = 'AdemolaCNN'
        custom_objects = {}
        cmEpoch, loss, acc = trainKerasModel(model=model, batchSize=batchSize,
                                             numEpochs=numEpochs, 
                                             sequences=[struct[0]], classes = struct[1], trainRange=struct[2],
                                             valRange = struct[3], testRange = struct[4],
                                             numClasses = struct[5], 
                                             modelFile = modelFile, callbacks = [EarlyStopping(patience=20)],
                                             sampleWeights=None, 
                                             outDirectory=outDirectory, trainMode=trainMode2,
                                             custom_objects = custom_objects,
                                             loss_function = 'categorical_crossentropy',
                                             optimizer = 'adam')
        #catch our breath.... Really, give the user a chance to insert Ctrl-C
        time.sleep(2)
        losses += [loss]
        accs += [acc]
        cmEpochs += [cmEpoch]
    losses = np.asarray(losses)
    accs = np.asarray(accs)*100
    trainPer, valPer, _, _ = normalizePercentages(trainPer, valPer, 0, 1)
    sys.stdout.write('\n')
    sys.stdout.write('**********\n')
    sys.stdout.write('Leave One User Out Evaluation, ' + modelFilePrefix + '-CDAN\n' 
                     +'Test Results for ' + outDirectory + '\n'
                     + str(numEpochs) + ' Maximum Epochs at ' + ("%0.2f" % trainPer) + '/' + ("%0.2f" % valPer) + ' Training/Validation Split\n')
    sys.stdout.write('\n')
    sys.stdout.write('Loss: ' + str(np.mean(losses)) + ' +/- ' + str(np.std(losses)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Loss: ' + str(np.percentile(losses, 25))
                     + ', ' + str(np.median(losses)) 
                     +  ', ' + str(np.percentile(losses, 75)) +'\n')
    sys.stdout.write('\n')
    sys.stdout.write('Accuracy: ' + str(np.mean(accs)) + ' +/- ' + str(np.std(accs)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Accuracy: ' + str(np.percentile(accs, 25))
                     + ', ' + str(np.median(accs)) 
                     +  ', ' + str(np.percentile(accs, 75)) +'\n')

def cdanTest(directory = defaultDirectory(),trainPer=0.6, 
             valPer=0.25, testPer=0.15, totalPer=1, 
             randSeed = 1, trainMode = 'continue',
             randSeed2 = None, numEpochs = 500,
             learningRate = 0.1, batchSize=32, patience = 20,
             trainAbs = 75, valAbs = 75, testAbs = 75,
             modelFilePrefix = '', prependMean = False,
             embeddingHiddenSize = 11,
             embeddingSize = 11, embeddingDepth = 2, 
             mlpHiddenSize = 11, mlpDepth = 2,
             maxout = 1, embeddingResidual = True,
             mlpResidual = False,
             poolingMode = 'ave',
             l2Reg = 0.001, gaussianNoise = 20,
             dropoutH = 0):
    """
    Evaluate effects of maxout and residual connections on CDAN peformance.
    """
    from keras.layers import Conv1D, Conv2D, Reshape, Permute, Activation, Flatten
    np.random.seed(randSeed)
    structs = loadDataset(directory=directory, delRange = None, trainPer = trainPer,
                         valPer=valPer, testPer=testPer, totalPer=totalPer, classAbbrv='c',
                         prune = False, classRange=range(1,6), preExt='.unw',
                         LOUO = True,
                         trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs)
    u=0
    losses = []
    accs = []
    cmEpochs = []
    
    outDirectory = 'CDAN'
    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)
    if randSeed2 is not None: #control randomization of training 
        np.random.seed(randSeed2)
    for struct in structs:
        modelFile = modelFilePrefix + 'LOU-' + str(u) +'-'
        modelFile = modelFile + outDirectory
        u += 1
        if (os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'.json') 
            and os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'_Weights.h5')):
            #if we have already trained for leaving out this user
            if trainMode == 'continue': #continue until each user has a model
                trainMode2 = 'skip' 
            elif trainMode == 'continue-each': # continue training previous models
                trainMode2 = 'continue'
            else:
                trainMode2 = 'overwrite'
        else:
            trainMode2 = trainMode
        struct = list(struct)
        """
        Sort markers by position.
        """
        struct[0] = sortData(struct[0].transpose([1,0,2]), np.eye(3), True).transpose([1,0,2])
        """
        Center all frames on their mean.
        """
        struct[0] = centerData(struct[0].transpose([1,0,2]), True, prependMean).transpose([1,0,2])
        """
        Normalize markers by position.
        """
        #struct[0] = normalizeData(struct[0].transpose([1,0,2]), True).transpose([1,0,2])
        """
         DANs only outputs a single class per sequence rather than an output for
         each timestep of the sequence. Fix the target class ID matrix to account
         for this.
         Could something similar be done for the RNNs? Yes!
        """
        struct[1] = struct[1][0,:].reshape((1, struct[7]))
        
        maxNumMarkers = struct[6] + (1 if prependMean else 0)
        numClasses = struct[5]
        inputLayer = Input(shape=(maxNumMarkers,3))
        model = Masking(0)(inputLayer)
        model = GaussianNoise(gaussianNoise)(model)

        #make embedding function, which will be time-distributed
        singleInput = Input(shape=(3,))
        kwargs = {'W_regularizer': l2(l2Reg)} #for residual connections
        if embeddingResidual == 'dense':
            embeddingLayers = []
            #create each layer separately
            for i in xrange(1, embeddingDepth+1):
                inputDim = 3 if i == 1 else embeddingHiddenSize
                inp = Input(shape=(inputDim,))
                embeddingLayer = addMaxoutLayer(inp, 
                                                Dense(output_dim = maxout*(embeddingHiddenSize if i < embeddingDepth else embeddingSize),
                                                      activation = 'relu' if maxout == 1 else 'linear',
                                                      W_regularizer = l2(l2Reg)), 
                                                maxout)
                if dropoutH and i < embeddingDepth:
                    embeddingLayer = Dropout(dropoutH)(embeddingLayer)
                embeddingLayer = Model(inp, embeddingLayer)
                embeddingLayers += [embeddingLayer]
            embedding = addDenseResidualLayers(singleInput, embeddingLayers, False, **kwargs)
        else:
            embedding = singleInput
            for i in xrange(1, embeddingDepth+1):
                embedding = addMaxoutLayer(embedding, 
                                           Dense(output_dim = maxout*(embeddingHiddenSize if i < embeddingDepth else embeddingSize),
                                                 activation = 'relu' if maxout == 1 else 'linear',
                                                 W_regularizer = l2(l2Reg)), 
                                           maxout)
                if dropoutH and i < embeddingDepth:
                    embedding = Dropout(dropoutH)(embedding)
            if embeddingResidual: #should make it an option to have residual connections every n layer(s)
                embedding = Model(input = singleInput, output = embedding)
                embedding = addResidualLayer(singleInput, embedding, **kwargs)
        if dropoutH:
            embedding = Dropout(dropoutH)(embedding)
        embedding = Model(input = singleInput, output = embedding)
        model = TimeDistributed(embedding)(model)
        
        #deep averaging network
        model = MaskEatingLambda(TimeDistributedMerge(isMasked = True, mode = poolingMode), 
                                 output_shape = lambda shape: (shape[0],)+shape[2:])(model)
        singleInput = Input(shape=(embeddingSize,))
        kwargs = {'W_regularizer': l2(l2Reg)} #for residual connections
        if mlpResidual == 'dense':
            mlpLayers = []
            #create each layer separately
            for i in xrange(1, mlpDepth+1):
                inputDim = embeddingSize if i == 1 else mlpHiddenSize
                inp = Input(shape=(inputDim,))
                mlpLayer = addMaxoutLayer(inp, 
                                          Dense(output_dim = maxout*mlpHiddenSize,
                                                activation = 'relu' if maxout == 1 else 'linear',
                                                W_regularizer = l2(l2Reg)), 
                                          maxout)
                if dropoutH and i < mlpDepth:
                    mlpLayer = Dropout(dropoutH)(mlpLayer)
                mlpLayer = Model(inp, mlpLayer)
                mlpLayers += [mlpLayer]
            mlp = addDenseResidualLayers(singleInput, mlpLayers, False, **kwargs)
        else:
            mlp = singleInput
            for i in xrange(1, mlpDepth):
                mlp = addMaxoutLayer(mlp, 
                                    Dense(output_dim = maxout*mlpHiddenSize,
                                            activation = ('relu' if maxout == 1 else 'linear'),
                                            W_regularizer = l2(l2Reg)), 
                                    maxout if i < mlpDepth else 1)
                if dropoutH and i < mlpDepth:
                    mlp = Dropout(dropoutH)(mlp)
            if mlpResidual:
                mlp = Model(input = singleInput, output = mlp)
                mlp = addResidualLayer(singleInput, mlp, **kwargs)
        mlp = Dense(output_dim = numClasses, activation = 'softmax', 
                    W_regularizer = l2(l2Reg))(mlp)
        mlp = Model(input = singleInput, output = mlp)
        model = mlp(model)

        model = Model(input = inputLayer, output = model)
        if modelFile is None:
            modelFile = 'CDAN'
        custom_objects = {'MaskEatingLambda': MaskEatingLambda, 'WordDropout': WordDropout, 'Reverse': Reverse, 
                          'Sort': Sort, 'MaskedPooling': MaskedPooling}
        cmEpoch, loss, acc = trainKerasModel(model=model, batchSize=batchSize,
                                             numEpochs=numEpochs, 
                                             sequences=[struct[0]], classes = struct[1], trainRange=struct[2],
                                             valRange = struct[3], testRange = struct[4],
                                             numClasses = struct[5], 
                                             modelFile = modelFile, callbacks = [EarlyStopping(patience=patience)],
                                             sampleWeights=None, 
                                             outDirectory=outDirectory, trainMode=trainMode2,
                                             custom_objects = custom_objects,
                                             loss_function = 'categorical_crossentropy',
                                             optimizer = RMSprop(learningRate))
        #catch our breath.... Really, give the user a chance to insert Ctrl-C
        time.sleep(2)
        losses += [loss]
        accs += [acc]
        cmEpochs += [cmEpoch]
    losses = np.asarray(losses)
    accs = np.asarray(accs)*100
    trainPer, valPer, _, _ = normalizePercentages(trainPer, valPer, 0, 1)
    sys.stdout.write('\n')
    sys.stdout.write('**********\n')
    sys.stdout.write('Leave One User Out Evaluation, ' + modelFilePrefix + '-CDAN\n' 
                     +'Test Results for ' + outDirectory + '\n'
                     + str(numEpochs) + ' Maximum Epochs at ' + ("%0.2f" % trainPer) + '/' + ("%0.2f" % valPer) + ' Training/Validation Split\n')
    sys.stdout.write('\n')
    sys.stdout.write('Loss: ' + str(np.mean(losses)) + ' +/- ' + str(np.std(losses)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Loss: ' + str(np.percentile(losses, 25))
                     + ', ' + str(np.median(losses)) 
                     +  ', ' + str(np.percentile(losses, 75)) +'\n')
    sys.stdout.write('\n')
    sys.stdout.write('Accuracy: ' + str(np.mean(accs)) + ' +/- ' + str(np.std(accs)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Accuracy: ' + str(np.percentile(accs, 25))
                     + ', ' + str(np.median(accs)) 
                     +  ', ' + str(np.percentile(accs, 75)) +'\n')

def autoencoderTest(directory = defaultDirectory(), trainPer=0.6, 
            valPer=0.25, testPer=0.15, totalPer=1, 
            randSeed = 1, trainMode = 'continue',
            randSeed2 = None, numEpochs = 500,
            learningRate = 0.1, batchSize=32,
            trainAbs = 75, valAbs = 75, testAbs = 75,
            modelFilePrefix = '', prependMean = False):
    """
    Train autoencoders.
    """
    from keras.layers import Conv1D, Conv2D, Reshape, Permute, Activation, Flatten
    np.random.seed(randSeed)
    structs = loadDataset(directory=directory, delRange = None, trainPer = trainPer,
                         valPer=valPer, testPer=testPer, totalPer=totalPer, classAbbrv='c',
                         prune = False, classRange=range(1,6), preExt='.unw',
                         LOUO = True,
                         trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs)
    u=0
    losses = []
    accs = []
    
    outDirectory = 'Autoencoders'
    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)
    if randSeed2 is not None: #control randomization of training 
        np.random.seed(randSeed2)
    for struct in structs:
        modelFile = modelFilePrefix + 'LOU-' + str(u) +'-'
        modelFile = modelFile + outDirectory
        weightsFile = modelFile+'_Weights'
        u += 1
        if (os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'.json') 
            and os.path.isfile(outDirectory + '\\' + 'Keras'+modelFile+'_Weights.h5')):
            #if we have already trained for leaving out this user
            if trainMode == 'continue': #continue until each user has a model
                trainMode2 = 'skip' 
            elif trainMode == 'continue-each': # continue training previous models
                trainMode2 = 'continue'
            else:
                trainMode2 = 'overwrite'
        else:
            trainMode2 = trainMode
        struct = list(struct)
        """
        Sort markers by position.
        """
        struct[0] = sortData(struct[0].transpose([1,0,2]), np.eye(3), True).transpose([1,0,2])
        """
        Center all frames on their mean.
        """
        struct[0] = centerData(struct[0].transpose([1,0,2]), True, prependMean).transpose([1,0,2])
        """
        Normalize markers by position.
        """
        #struct[0] = normalizeData(struct[0].transpose([1,0,2]), True).transpose([1,0,2])
        """
         DANs only outputs a single class per sequence rather than an output for
         each timestep of the sequence. Fix the target class ID matrix to account
         for this.
         Could something similar be done for the RNNs? Yes!
        """
        struct[1] = struct[1][0,:].reshape((1, struct[7]))
        
        maxNumMarkers = struct[6] + (1 if prependMean else 0)
        numClasses = struct[5]
        inputLayer = Input(shape=(maxNumMarkers,3))
        #make encoder
        encoder = Masking(0)(inputLayer)
        encoder = GaussianNoise(20)(encoder)
        encoder = TimeDistributed(Dense(output_dim = 33, activation = 'relu', W_regularizer = l2(0.01)))(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = TimeDistributed(Dense(output_dim = 33, activation = 'relu', W_regularizer = l2(0.01)))(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = MaskEatingLambda(TimeDistributedMerge(isMasked = True, mode = 'max'), 
                             output_shape = lambda shape: (shape[0],)+shape[2:])(encoder)
        encoder = Model(input = inputLayer, output = encoder)
        #make decoder (i.e. autoencoder)
        autoencoder = encoder(inputLayer)
        autoencoder = RepeatVector(maxNumMarkers)(autoencoder)
        autoencoder = GRU(66, return_sequences = True, 
                            W_regularizer= l2(0.01))(autoencoder)
        autoencoder = Dropout(0.1)(autoencoder)
        autoencoder = TimeDistributed(Dense(output_dim = 3, activation = 'linear'))(autoencoder)
        #encoder, autoencoder = makeAutoencoder(inputLayer, 12, 3, 11, 11, 1, True, 0.01, 0.1, 'relu', 0, 11, False)
        autoencoder = Model(inputLayer, autoencoder)
        autoencoder.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                      sample_weight_mode='temporal')

        trainingData = struct[0].transpose([1,0,2])[struct[2], :, :]
        validationData = struct[0].transpose([1,0,2])[struct[3], :, :]
        testData = struct[0].transpose([1,0,2])[struct[4], :, :]

        autoencoder.fit(x = trainingData, y = trainingData,
                  sample_weight = getMask(trainingData, keepdims = False),
                  validation_data = (validationData, validationData, getMask(validationData,keepdims = False)),
                  batch_size = batchSize, nb_epoch = numEpochs, verbose = 2,
                  callbacks = [ModelCheckpoint(weightsFile + '.h5', save_best_only = True), EarlyStopping(patience = 20)])
        loss = autoencoder.test_on_batch(x=testData, y=testData, sample_weight = getMask(testData, keepdims = False))
        print "Test loss: " + str(loss)
        # use learned encoding for classification
        if os.path.isfile(weightsFile+'.h5'):
            autoencoder.load_weights(weightsFile+'.h5') #does this also load the weights into encoder?
        classifier = encoder(inputLayer)
        classifier = Dense(output_dim = 5, activation = 'softmax')(classifier)
        classifier = Model(input = inputLayer, output = classifier)
        custom_objects = {'MaskEatingLambda': MaskEatingLambda, 'WordDropout': WordDropout, 'Reverse': Reverse, 
                          'Sort': Sort, 'MaskedPooling': MaskedPooling}
        cmEpoch, loss, acc = trainKerasModel(model=classifier, batchSize=batchSize,
                                             numEpochs=numEpochs, 
                                             sequences=[struct[0]], classes = struct[1], trainRange=struct[2],
                                             valRange = struct[3], testRange = struct[4],
                                             numClasses = struct[5], 
                                             modelFile = modelFile, callbacks = [EarlyStopping(patience=20)],
                                             sampleWeights=None, 
                                             outDirectory=outDirectory, trainMode=trainMode2,
                                             custom_objects = custom_objects,
                                             loss_function = 'categorical_crossentropy',
                                             optimizer = 'rmsprop')
        losses += [loss]
        #catch our breath.... Really, give the user a chance to insert Ctrl-C
        time.sleep(2)
        losses += [loss]
        accs += [acc]
    losses = np.asarray(losses)
    accs = np.asarray(accs)*100
    trainPer, valPer, _, _ = normalizePercentages(trainPer, valPer, 0, 1)
    sys.stdout.write('\n')
    sys.stdout.write('**********\n')
    sys.stdout.write('Leave One User Out Evaluation, ' + modelFilePrefix + '-AdemolaCNN\n' 
                     +'Test Results for ' + outDirectory + '\n'
                     + str(numEpochs) + ' Maximum Epochs at ' + ("%0.2f" % trainPer) + '/' + ("%0.2f" % valPer) + ' Training/Validation Split\n')
    sys.stdout.write('\n')
    sys.stdout.write('Loss: ' + str(np.mean(losses)) + ' +/- ' + str(np.std(losses)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Loss: ' + str(np.percentile(losses, 25))
                     + ', ' + str(np.median(losses)) 
                     +  ', ' + str(np.percentile(losses, 75)) +'\n')
    sys.stdout.write('\n')

if __name__ == '__main__':
    # catching ctrl-c wasn't working in Windows cmd prompt. 
    # Some problem with scipy, fortran library, other stuff behind scenes.
    if os.name == 'nt':
        import thread
        import win32api
        def ctrlCHandler(dwCtrlType, hook_sigint=thread.interrupt_main):
            if dwCtrlType == 0: # CTRL-C Event
                hook_sigint()
                return True #don't chain to next handler
            return False
        win32api.SetConsoleCtrlHandler(ctrlCHandler, 1)
    #comprehensivePostEvaluation(trainMode = 'continue', numEpochs = 10)
    
    #autoencoderTest(trainMode = 'overwrite')
    ##embeddingHiddenSize = 11
    ##embeddingSize = 100
    ##embeddingDepth = 2
    ##embeddingResidual = False
    ##mlpHiddenSize = 11
    ##mlpDepth = 2
    ##mlpResidual = 'dense'
    ##maxout = 2
    ##l2Reg = 0.001
    ##dropoutH = 0.1
    ##poolingMode = 'sum'
    ##gaussianNoise = 20
    ##patience = 40
    ##modelFilePrefix = '-'.join(['L', str(embeddingDepth), str(mlpDepth), 
    ##                            'N', str(embeddingHiddenSize), str(embeddingSize), str(mlpHiddenSize),
    ##                            'PM', poolingMode,
    ##                            (('maxout' + str(maxout)) if maxout > 1 else 'relu'),
    ##                            'l2', str(l2Reg), 
    ##                            'GN', str(gaussianNoise),
    ##                            'D', str(dropoutH),
    ##                            'res', 
    ##                            '' if not embeddingResidual else ('dE' if embeddingResidual == 'dense' else 'E'), 
    ##                            '' if not mlpResidual else ('dD' if mlpResidual == 'dense' else 'D')])
    ##cdanTest(learningRate = 0.001, trainMode = 'continue', modelFilePrefix = modelFilePrefix, numEpochs = 500, prependMean = False, 
    ##         embeddingHiddenSize = embeddingHiddenSize, 
    ##         embeddingSize = embeddingSize, embeddingDepth = embeddingDepth, embeddingResidual = embeddingResidual,
    ##         mlpHiddenSize = mlpHiddenSize, mlpDepth = mlpDepth, mlpResidual = mlpResidual,
    ##         maxout = maxout, l2Reg = l2Reg, dropoutH = dropoutH, 
    ##         poolingMode = poolingMode, gaussianNoise = gaussianNoise,
    ##         patience = patience)
    #cnnTest(learningRate = 0.01, trainMode = 'continue', modelFilePrefix = 'MLPCenteredPGNL2', numEpochs = 500, prependMean = True, justMLP = True)
    #cnnTest(learningRate = 0.01, trainMode = 'continue', modelFilePrefix = 'MLPCenteredGNL2', numEpochs = 500, prependMean = False, justMLP = True)
    kwargs = [dict(), dict(), dict(), dict()]
    kwargs[0]['modelType'] = 'cdan'
    kwargs[1]['modelType'] = 'birdan'
    kwargs[2]['modelType'] = 'birnn'
    kwargs[3]['modelType'] = 'pmeq'
    # (mostly) shared properties
    for kwarg in kwargs:
        kwarg['masking'] = True
        kwarg['center'] = True
        kwarg['prependMean'] = False
        kwarg['numSpecialLayers'] = 2
        kwarg['numSpecialOut'] = 11
        kwarg['activation'] = 'relu'
        kwarg['maxout'] = 1
        kwarg['deepResidual'] = 'dense'
        kwarg['poolingMode'] = 'ave'
        kwarg['numDeepLayers'] = 2
        kwarg['useGRU'] = True
        kwarg['numSpecialNodes'] = 11
        kwarg['dropoutH'] = 0.1
        kwarg['l2Reg'] = 0.001
        kwarg['gaussianNoise'] = 20
        kwarg['numDeepNodes'] = 11
        kwarg['wordDropout'] = 0
    #add maxout activation variants
    kwargs = kwargs + [dict(kwarg) for kwarg in kwargs]
    for i in xrange(len(kwargs)/2, len(kwargs)):
        kwargs[i]['maxout'] = 2
        kwargs[i]['activation'] = 'linear'
    #add pooling modes for cdan and birdan
    for i in range(len(kwargs)):
        if kwargs[i]['modelType'] != 'birnn':
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = 'max'
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = 'sum'
        if kwargs[i]['modelType'] == 'pmeq':
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = ['sum', 'max']
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = ['max', 'sum']
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = ['ave', 'max']
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = ['max', 'ave']
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = ['ave', 'sum']
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['poolingMode'] = ['sum', 'ave']
    #add larger embeddings for CDAN
    for i in range(len(kwargs)):
        if kwargs[i]['modelType'] == 'cdan' or kwargs[i]['modelType'] == 'pmeq':
            kwargs += [dict(kwargs[i])]
            kwargs[-1]['numSpecialOut'] = 100
    #add linear embedding CDAN with max pooling
    kwargs += [dict(kwargs[0])]
    kwargs[-1]['modelType'] = 'cdan'
    kwargs[-1]['numSpecialLayers'] = 1
    kwargs[-1]['poolingMode'] = 'max'
    kwargs[-1]['activation'] = 'linear'
    kwargs[-1]['maxout'] = [1, 2]
    #add larger linear embedding with max pooling
    kwargs += [dict(kwargs[-1])]
    kwargs[-1]['numSpecialOut'] = 100
    #add no embedding with sum pooling (equivalent to average pooling)
    kwargs += [dict(kwargs[-1])]
    kwargs[-1]['numSpecialLayers'] = 0
    kwargs[-1]['poolingMode'] = 'sum'
    #split on number of deep layers
    #kwargs = kwargs + [dict(kwarg) for kwarg in kwargs]
    #for i in xrange(len(kwargs)):
    #    kwargs[i]['numDeepLayers'] = 2 if i < len(kwargs)/2 else 3
    
    #remove all but pmeq models with maxout activation
    kwargs = [kwarg for kwarg in kwargs if kwarg['modelType'] == 'pmeq']
    #remove all but pmeq models with maxout activation
    kwargs = [kwarg for kwarg in kwargs if kwarg['modelType'] == 'pmeq' and kwarg['activation'] == 'linear']
    #remove all but pmeq models with maxout activation and smaller embeddings
    kwargs = [kwarg for kwarg in kwargs if kwarg['modelType'] == 'pmeq' and kwarg['activation'] == 'linear' and kwarg['numSpecialOut'] == 11]
    #kwargs = [kwarg for kwarg in kwargs if (kwarg['modelType'] == 'pmeq' 
    #                                        and kwarg['activation'] == 'linear' 
    #                                        and kwarg['numSpecialOut'] == 11 
    #                                        and ((isinstance(kwarg['poolingMode'], list) and kwarg['poolingMode'][0] == 'ave')
    #                                             or kwarg['poolingMode'] == 'ave'))]
    #kwargs = [kwarg for kwarg in kwargs if (isinstance(kwarg['poolingMode'], list) and kwarg['poolingMode'][1] == 'max')]
    #get just the basic sum, ave, and max pooling modes
    kwargs = [kwarg for kwarg in kwargs if not isinstance(kwarg['poolingMode'], list)]
    #change model type to fpmeq
    for i in range(len(kwargs)):
        kwargs[i]['modelType'] = 'cdan'
        kwargs += [dict(kwargs[i])]
        kwargs += [dict(kwargs[i])]
        kwargs += [dict(kwargs[i])]
        kwargs[-2]['numSpecialOut'] = 100
        kwargs[-1]['numSpecialOut'] = 100
        kwargs[-3]['deepResidual'] = False
        kwargs[-1]['deepResidual'] = False

    kwargs = [kwarg for kwarg in kwargs if kwarg['deepResidual'] == False]

    numTrials = 5
    seeds = range(1, numTrials+1)
    justResults = False

    allMetrics = dict()
    for i in xrange(0, len(kwargs)):
        kwarg = kwargs[i]
        if kwarg['modelType'] == 'cdan':
            key = namePostureModel(randSeed = 0, trainAbs = 75, valAbs = 75, testAbs = 75, **kwarg)
            allMetrics[key] = {'args': kwarg, 'losses': [], 'accuracies': []}
            for seed in seeds:
                _, losses, accs = comprehensivePostLOOEvaluation(randSeed = seed,  
                                trainMode = 'continue',
                                learningRate = 0.001, patience = 40,
                                saveResults = True,
                                justResults = justResults, **kwarg)
                allMetrics[key]['losses'] += [losses]
                allMetrics[key]['accuracies'] += [accs]
                if len(losses) != 12 or len(accs) != 12:
                    warnings.warn("Trials missing: " + key)
                    allMetrics[key]['losses'].pop()
                    allMetrics[key]['accuracies'].pop()

    for modelType in ['cdan']: #['cdan', 'birdan', 'birnn']:
        allKeys = []
        allLossMeans = []
        allLossStds = []
        allAccMeans = []
        allAccStds = []
        for key, value in allMetrics.iteritems():
            if value['args']['modelType'] == modelType and len(value['losses']) > 0:
                losses = np.asarray(value['losses'])
                accuracies = np.asarray(value['accuracies'])
                losses = np.concatenate([losses, np.mean(losses, axis = 1, keepdims = True)], axis = 1)
                accuracies = np.concatenate([accuracies, np.mean(accuracies, axis = 1, keepdims = True)], axis = 1)
                lossMeans = np.mean(losses, axis = 0)
                lossStds = np.std(losses, axis = 0)
                accMeans = np.mean(accuracies, axis = 0)
                accStds = np.std(accuracies, axis = 0)
                sys.stdout.write('\r\n\r\n')
                sys.stdout.write(key + '\r\n')
                sys.stdout.write('Average losses: \r\n\t' + str(lossMeans) + '\r\n\t +/- \r\n\t' + str(lossStds))
                sys.stdout.write('\r\n')
                sys.stdout.write('Average accuracies: \r\n\t' + str(accMeans) + '\r\n\t +/- \r\n\t' + str(accStds))
                sys.stdout.write('\r\n\r\n')
                allKeys += [key]
                allLossMeans += [lossMeans]
                allLossStds += [lossStds]
                allAccMeans += [accMeans]
                allAccStds += [accStds]
        allLossMeans = np.asarray(allLossMeans).T
        allLossStds = np.asarray(allLossStds).T
        allAccMeans = np.asarray(allAccMeans).T
        allAccStds = np.asarray(allAccStds).T
        sys.stdout.write('CONDENSED INFO\r\n')
        sys.stdout.write(modelType + ' Keys: \r\n')
        for key in allKeys:
            sys.stdout.write(key + '\r\n')
        def printTable(means, stds, modelType, measure):
            sys.stdout.write('LaTeX FORMATTED TABLE COLUMNS FOR TYPE ' + modelType + ' ' + measure + '\r\n')
            for i in xrange(means.shape[0]):
                for j in xrange(means.shape[1]):
                    entry = '\\underset{\\pm '+"{0:.2f}".format(stds[i,j])+'}{'+"{0:.2f}".format(means[i,j])+'}'
                    if j > 0:
                        sys.stdout.write(' & ')
                    sys.stdout.write(entry)
                sys.stdout.write(' \\\\\r\n')
            sys.stdout.write('\r\n\r\n')
        printTable(allLossMeans, allLossStds, modelType, 'LOSSES')
        printTable(allAccMeans, allAccStds, modelType, 'ACCURACIES')

    #for masking in [True]:
    #    for modelType in ['cdan']: #['birdan', 'birnn']: #
    #        for prependMean in [False]: #[True, False]:
    #            for numDeepLayers in [2,3]:
    #                for poolingMode in ['max']:
    #                    comprehensivePostLOOEvaluation(modelType = modelType, l2Reg = 0.001, 
    #                                    gaussianNoise = 20, wordDropout = 0, 
    #                                    useGRU=True, trainMode = 'overwrite',
    #                                    poolingMode = poolingMode, numSpecialLayers = 1, 
    #                                    numSpecialOut = 100, numSpecialNodes = 11,
    #                                    learningRate = 0.001, patience = 40,
    #                                    numDeepLayers = numDeepLayers, numDeepNodes = 11,
    #                                    activation = 'linear', masking = masking,
    #                                    dropoutH = 0.1, center = True, prependMean = prependMean,
    #                                    maxout = [1, 2], deepResidual = 'dense')

    #masks = [True, False]
    #noises = [0, 10, 20, 30]
    #dropoutHs = [0, 0.5]
    #l2Regs = [0, 0.001, 0.01, 0.1]
    #for mask in masks:
    #    for dropoutH in dropoutHs:
    #        for noise in noises:
    #            comprehensivePostLOOEvaluation(modelType = 'birnn', l2Reg = 0.01, 
    #                            gaussianNoise = noise, wordDropout = 0, 
    #                            useGRU=True, trainMode = 'continue',
    #                            poolingMode = 'max', numSpecialLayers = 2, 
    #                            numSpecialOut = 11, numSpecialNodes = 11,
    #                            learningRate = 0.001, 
    #                            numDeepLayers = 3, numDeepNodes = 11,
    #                            activation = 'relu', masking = mask,
    #                            dropoutH = dropoutH)