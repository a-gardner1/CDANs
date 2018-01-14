
import time
import os
import sys
import numpy as np
from keras.layers import merge, Input
from keras.models import Model, model_from_json
from keras.layers.core import Masking, Dense, Reshape, Dropout, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import rmsprop
from Postures import buildPostureModel, namePostureModel, sortData, centerData
from Gestures import loadDataset, normalizePercentages
from KerasSupplementary import trainKerasModel, addRealGaussianNoise
from KerasSupplementary import WordDropout, Reverse, MaskEatingLambda, Sort, MaskedPooling
import warnings

"""
Plan for how to tackle this since Keras does not support multidimensional masks and RNNs.

1.  Load each frame, padding the number of markers with zeros 
    to match the frame with the most markers and padding each sequence of frames
    to match the longest sequence.
    This results in a (numSamples, numTimeSteps, numFeatures) shaped array.
2.  Split the input into labeled and unlabeled features.
3.  Reshape the unlabeled features to be of shape (numSamples, numTimeSteps, maxNumMarkers, 3).
4.  Apply a Masking(0) layer separately to each set of features.
5.  Provide each input to its respective part of the planned network.
6.  Use a TimeDistributed wrapper on the unlabeled model (hope that masking works as expected).
7.  Concatenate the labeled and unlabeled outputs. The mask for unlabeled features should have been consumed
    as part of its internal network (the outputs' dimensionality is 1 less than the input). The mask for
    the labeled features will then be propagated to the rest of the network.

Masking works based upon the current implementation of Keras (version 1.0.7). 
Older versions will break this code, newer versions might.
"""

def defaultDirectory():
    return '.'

def makeSingleOutputModel(unlabeledModel, maxNumMarkers, totalNumFeatures, 
                          numTimeSteps, numClasses, useGRU = False,
                          dropoutI = 0, dropoutH = 0, numRNNLayers = 2, numRNNNodes = 200,
                          numDeepLayers = 1, numDeepNodes = 100, activation = 'relu',
                          l2Reg = 0.0001, prependedMean = False):
    """
    Assumption: 
    unlabeledModel is completely specified from input to output, including masking, dropout, and noise.
    The unlabeledModel assumes markers are given sequentially as input.
    The unlabeledModel is a type derived from Keras Layer.

    numDeepLayers does not include the softmax output layer.
    """
    labeledInput = Input(shape=(numTimeSteps, totalNumFeatures-3*maxNumMarkers))
    unlabeledInput = Input(shape=(numTimeSteps, maxNumMarkers+(1 if prependedMean else 0), 3))
    labeledModel = Masking(0)(labeledInput)
    if dropoutI:
        labeledModel = Dropout(dropoutI)(labeledModel)
    #let totalModel start out as the complete unlabeledModel
    totalModel = TimeDistributed(unlabeledModel)(unlabeledInput)
    totalModel = merge([labeledModel, totalModel], mode = 'concat')
    for i in range(numRNNLayers):
        if useGRU:
            totalModel = GRU(output_dim = numRNNNodes, W_regularizer=l2(l2Reg), 
                             return_sequences=True)(totalModel)
        else:
            totalModel = LSTM(output_dim = numRNNNodes, W_regularizer=l2(l2Reg), 
                             return_sequences=True)(totalModel)
        if dropoutH:
            totalModel = Dropout(dropoutH)(totalModel)
    for i in range(numDeepLayers):
        totalModel = TimeDistributed(Dense(output_dim=numDeepNodes, W_regularizer = l2(l2Reg), 
                                           activation=activation))(totalModel)
        if dropoutH:
            totalModel = Dropout(dropoutH)(totalModel)
    totalModel = TimeDistributed(Dense(output_dim=numClasses, W_regularizer = l2(l2Reg),
                                       activation = 'softmax'))(totalModel)
    return Model(input = [labeledInput, unlabeledInput], output = totalModel)

def comprehensiveGestEvaluation(directory=defaultDirectory(), trainPer=0.6, 
                                valPer=0.25, testPer=0.15, totalPer=1, 
                                randSeed = 1, modelType = 'cdan', 
                                numSpecialLayers = 2, numULDeepLayers = 2,
                                numSpecialNodes = 11, numULDeepNodes = 11,
                                numSpecialOut = 11, numULOut = 11,
                                useGRU=True, l2Reg = 0.0001,
                                ULl2Reg = .0001, activation='relu',
                                dropoutI=0, dropoutH=0,
                                batchSize=64, numEpochs = 500, 
                                learningRate = 0.001, trainMode = 'continue',
                                modelFile = None, 
                                trainAbs = None, valAbs = None, testAbs = None,
                                augmentData = None, wordDropout=0,
                                gaussianNoise = 0, 
                                numRNNLayers = 2, numRNNNodes = 200,
                                numDeepLayers = 1, numDeepNodes = 100, 
                                pruneGlobal = True, makeSymmetric = False,
                                ULPoolingMode = 'ave', ULMasking = True,
                                ULCenter = True, prependMean = True):
    np.random.seed(randSeed)
    augmentData = None #override until data augmentation is implemented
    warnings.warn("Parameter 'augmentData' has no effect in the current implementation")

    trainPer, valPer, testPer, totalPer = normalizePercentages(trainPer, valPer, testPer, totalPer)
    
    # prune global coordinate data?
    if pruneGlobal:
        pruneRange = range(0, 18)
        numLabeledFeatures = 18
    else:
        pruneRange = None
        numLabeledFeatures = 36
    
    struct = loadDataset(directory, pruneRange, trainPer, valPer, 
                         testPer, totalPer, '.left', True)
    numClasses = struct[5]
    numTimeSteps = struct[6]
    numSamples = struct[7]
    totalNumFeatures = struct[8]
    maxNumMarkers = (totalNumFeatures - numLabeledFeatures)/3
    labeledData = struct[0][:, :, 0:numLabeledFeatures]
    unlabeledData = struct[0][:, :, numLabeledFeatures:totalNumFeatures].reshape((numTimeSteps, numSamples, maxNumMarkers, 3))

    """
    Sort markers by position.
    """
    unlabeledData = sortData(unlabeledData.reshape((-1, maxNumMarkers, 3)), 
                         np.eye(3), True).reshape((numTimeSteps, numSamples, maxNumMarkers, 3))
    """
    Center each frame on its mean.
    """
    if ULCenter:
        unlabeledData = centerData(unlabeledData.reshape((-1, maxNumMarkers, 3)), True, prependMean).reshape((numTimeSteps, numSamples, maxNumMarkers, 3))

    #make the unlabeled model
    inputLayer = Input(shape=(maxNumMarkers+(1 if prependMean else 0), 3))
    model = inputLayer
    if wordDropout:
        model = WordDropout(wordDropout, False)(model)
    if ULMasking:
        model = Masking(0)(model)
    if gaussianNoise:
        model = addRealGaussianNoise(model, gaussianNoise, ULMasking)
    if dropoutI:
        model = Dropout(dropoutI)(model)

    model = buildPostureModel(input = model, modelType=modelType, numSpecialLayers=numSpecialLayers,
                              numDeepLayers=numULDeepLayers, numSpecialNodes=numSpecialNodes,
                              numDeepNodes=numULDeepNodes, numSpecialOut=numSpecialOut,
                              useGRU=useGRU, l2Reg=ULl2Reg, 
                              dropoutH=dropoutH, activation=activation,
                              numClasses=numULOut,
                              makeSymmetric = makeSymmetric, stripSoftmax = True, 
                              poolingMode = ULPoolingMode, masking = ULMasking)
    #must formally make it into a model so that it can technically be a Layer
    model = Model(input = inputLayer, output = model)
    if modelFile is None:
        unlabeledModelFile = namePostureModel(randSeed,modelType=modelType, numSpecialLayers=numSpecialLayers,
                                     numDeepLayers=numULDeepLayers, numSpecialNodes=numSpecialNodes,
                                     numDeepNodes=numULDeepNodes, numSpecialOut=numSpecialOut,
                                     useGRU=useGRU, l2Reg=ULl2Reg, dropoutI=dropoutI,
                                     dropoutH=dropoutH, activation=activation,
                                     trainPer=trainPer, valPer=valPer, testPer=testPer,
                                     trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs,
                                     augmentData=augmentData, 
                                     wordDropout = wordDropout, gaussianNoise = gaussianNoise,
                                     makeSymmetric = makeSymmetric, poolingMode = ULPoolingMode,
                                     masking = ULMasking, center = ULCenter, prependMean = prependMean)
        modelFile = nameModelFile('', useGRU, unlabeledModelFile, 
                                  numRNNLayers, numRNNNodes, 
                                  numDeepLayers, numDeepNodes, 
                                  numULOut, l2Reg, numULOut)
    #make the complete model
    model = makeSingleOutputModel(model, maxNumMarkers, totalNumFeatures, 
                                  numTimeSteps, numClasses, useGRU=useGRU, dropoutI=dropoutI,
                                  dropoutH=dropoutH, numRNNLayers = numRNNLayers, numRNNNodes=numRNNNodes,
                                  numDeepLayers = numDeepLayers, numDeepNodes = numDeepNodes,
                                  activation = activation, l2Reg = l2Reg, 
                                  prependedMean = prependMean)
    custom_objects = {'MaskEatingLambda': MaskEatingLambda, 'WordDropout': WordDropout, 'Reverse': Reverse,
                      'Sort': Sort, 'MaskedPooling': MaskedPooling}

    cmEpoch, loss, acc, balAcc, finAcc = trainKerasModel(model=model, batchSize=batchSize,
                                                         numEpochs=numEpochs, 
                                                         sequences=[labeledData, unlabeledData], classes = struct[1], trainRange=struct[2],
                                                         valRange = struct[3], testRange = struct[4],
                                                         numClasses = numClasses, 
                                                         modelFile = modelFile, callbacks = [EarlyStopping(patience=20)],
                                                         sampleWeights=None, 
                                                         outDirectory='', trainMode=trainMode,
                                                         custom_objects = custom_objects,
                                                         optimizer = rmsprop(learningRate),
                                                         loss_function = 'categorical_crossentropy')

def comprehensiveGestLOOEvaluation(directory=defaultDirectory(), trainPer=0.6, 
                                    valPer=0.25, testPer=0.15, totalPer=1, 
                                    randSeed = 1, randSeed2 = None, modelType = 'cdan', 
                                    numSpecialLayers = 2, numULDeepLayers = 2,
                                    numSpecialNodes = 11, numULDeepNodes = 11,
                                    numSpecialOut = 11, numULOut = 11,
                                    useGRU=True, l2Reg = 0.0001,
                                    ULl2Reg = 0.0001, activation='relu',
                                    dropoutI=0, dropoutH=0,
                                    batchSize=64, numEpochs = 500, 
                                    learningRate = 0.001, trainMode = 'continue',
                                    modelFile = None, 
                                    trainAbs = None, valAbs = None, testAbs = None,
                                    augmentData = None, wordDropout=0,
                                    gaussianNoise = 0, 
                                    numRNNLayers = 2, numRNNNodes = 200,
                                    numDeepLayers = 1, numDeepNodes = 100, 
                                    pruneGlobal = True, makeSymmetric = False,
                                    modelFilePrefix = '', ULPoolingMode = 'ave',
                                    ULMasking = True, 
                                    ULCenter = True, prependMean = True):
    """
    Train RNNs for a leave-one-user-out evaluation.
    """
    
    trainModes = ['continue', 'overwrite', 'continue-each']
    augmentData = None #override until data augmentation is implemented
    warnings.warn("Parameter 'augmentData' has no effect in the current implementation")
    
    if trainMode.lower() not in trainModes:
        raise ValueError("Parameter 'trainMode' must be either 'continue', 'overwrite', or 'continue-each'.")
    
    np.random.seed(randSeed) #control permutation of data
    # prune global coordinate data?
    if pruneGlobal:
        pruneRange = range(0, 18)
        numLabeledFeatures = 18
    else:
        pruneRange = None
        numLabeledFeatures = 36
    structs = loadDataset(directory=directory, LOUO=True, 
                          delRange=pruneRange, trainPer=trainPer,
                          valPer = valPer, testPer=testPer, totalPer=totalPer,
                          preExt = '.left', prune=True)
    
    u=0
    losses = []
    accs = []
    balAccs = []
    finAccs = []
    cmEpochs = []
    unlabeledModelFile = namePostureModel(randSeed,modelType=modelType, numSpecialLayers=numSpecialLayers,
                                          numDeepLayers=numULDeepLayers, numSpecialNodes=numSpecialNodes,
                                          numDeepNodes=numULDeepNodes, numSpecialOut=numSpecialOut,
                                          useGRU=useGRU, l2Reg=ULl2Reg, dropoutI=dropoutI,
                                          dropoutH=dropoutH, activation=activation,
                                          trainPer=trainPer, valPer=valPer, testPer=testPer,
                                          trainAbs = trainAbs, testAbs = testAbs, valAbs=valAbs,
                                          augmentData=augmentData, 
                                          wordDropout = wordDropout, gaussianNoise = gaussianNoise,
                                          makeSymmetric = makeSymmetric, poolingMode = ULPoolingMode,
                                          masking = ULMasking, center = ULCenter, prependMean = prependMean)
    outDirectory = nameModelFile('', useGRU, unlabeledModelFile, 
                                 numRNNLayers, numRNNNodes, 
                                 numDeepLayers, numDeepNodes, 
                                 numULOut, l2Reg, numULOut)
    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)
    if randSeed2 is not None: #control randomization of training (for Keras at least)
        np.random.seed(randSeed2)
    for struct in structs:
        modelFile = modelFilePrefix + '-LOU-' + str(u) +'-'
        modelFile = modelFile
        u += 1
        if (os.path.isfile(outDirectory + '\\' + 'Keras' +modelFile+'.json') 
            and os.path.isfile(outDirectory + '\\' + 'Keras' +modelFile+'_Weights.h5')):
            #if we have already trained for leaving out this user
            if trainMode == 'continue': #continue until each user has a model
                trainMode2 = 'skip' 
            elif trainMode == 'continue-each': # continue training previous models
                trainMode2 = 'continue'
            else:
                trainMode2 = 'overwrite'
        else:
            trainMode2 = trainMode

        numClasses = struct[5]
        numTimeSteps = struct[6]
        numSamples = struct[7]
        totalNumFeatures = struct[8]
        maxNumMarkers = (totalNumFeatures - numLabeledFeatures)/3
        labeledData = struct[0][:, :, 0:numLabeledFeatures]
        unlabeledData = struct[0][:, :, numLabeledFeatures:totalNumFeatures].reshape((numTimeSteps, numSamples, maxNumMarkers, 3))
        
        """
        Sort markers by position.
        """
        unlabeledData = sortData(unlabeledData.reshape((-1, maxNumMarkers, 3)), 
                             np.eye(3), True).reshape((numTimeSteps, numSamples, maxNumMarkers, 3))
        """
        Center each frame on its mean.
        """
        if ULCenter:
            unlabeledData = centerData(unlabeledData.reshape((-1, maxNumMarkers, 3)), True, prependMean).reshape((numTimeSteps, numSamples, maxNumMarkers+(1 if prependMean else 0), 3))

        #make the unlabeled model; +1 for prependedMean
        inputLayer = Input(shape=(maxNumMarkers+(1 if prependMean else 0), 3))
        model = inputLayer
        if wordDropout:
            model = WordDropout(wordDropout, False)(model)
        if ULMasking:
            model = Masking(0)(model)
        if gaussianNoise:
            model = addRealGaussianNoise(model, gaussianNoise, ULMasking)
        if dropoutI:
            model = Dropout(dropoutI)(model)
        model = buildPostureModel(modelType=modelType, numSpecialLayers=numSpecialLayers,
                                  numDeepLayers=numULDeepLayers, numSpecialNodes=numSpecialNodes,
                                  numDeepNodes=numULDeepNodes, numSpecialOut=numSpecialOut,
                                  useGRU=useGRU, l2Reg=ULl2Reg,
                                  dropoutH=dropoutH, activation=activation,
                                  numClasses=numULOut, input = model,
                                  makeSymmetric = makeSymmetric, stripSoftmax = True, 
                                  poolingMode = ULPoolingMode, masking = ULMasking)
        model = Model(input = inputLayer, output = model)
        #make the complete model
        model = makeSingleOutputModel(model, maxNumMarkers, totalNumFeatures, 
                                      numTimeSteps, numClasses, useGRU=useGRU, dropoutI=dropoutI,
                                      dropoutH=dropoutH, numRNNLayers = numRNNLayers, numRNNNodes=numRNNNodes,
                                      numDeepLayers = numDeepLayers, numDeepNodes = numDeepNodes,
                                      activation = activation, l2Reg = l2Reg, 
                                      prependedMean = prependMean)
        custom_objects = {'MaskEatingLambda': MaskEatingLambda, 'WordDropout': WordDropout, 'Reverse': Reverse,
                          'Sort': Sort, 'MaskedPooling': MaskedPooling}
        cmEpoch, loss, acc, balAcc, finAcc = trainKerasModel(model=model, batchSize=batchSize,
                                             numEpochs=numEpochs,
                                             sequences=[labeledData, unlabeledData], classes = struct[1], trainRange=struct[2],
                                             valRange = struct[3], testRange = struct[4],
                                             numClasses = struct[5], 
                                             modelFile = modelFile, callbacks = [EarlyStopping(patience=20)],
                                             sampleWeights=None, 
                                             outDirectory=outDirectory, trainMode=trainMode2,
                                             custom_objects = custom_objects,
                                             optimizer = rmsprop(learningRate),
                                             loss_function = 'categorical_crossentropy')
        #catch our breath.... Really, give the user a chance to insert Ctrl-C
        time.sleep(2)
        losses += [loss]
        accs += [acc]
        balAccs += [balAcc]
        finAccs += [finAcc]
        cmEpochs += [cmEpoch]
    losses = np.asarray(losses)
    accs = np.asarray(accs)*100
    balAccs = np.asarray(balAccs)*100
    finAccs = np.asarray(finAccs)*100
    trainPer, valPer, _, _ = normalizePercentages(trainPer, valPer, 0, 1)
    sys.stdout.write('\n')
    sys.stdout.write('Leave One User Out Evaluation\nTest Results for ' + outDirectory + '\n'
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
    sys.stdout.write('\n')
    sys.stdout.write('Balanced Accuracy: ' + str(np.mean(balAccs)) + ' +/- ' + str(np.std(balAccs)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Balanced Accuracy: ' + str(np.percentile(balAccs, 25))
                     + ', ' + str(np.median(balAccs))
                     +  ', ' + str(np.percentile(balAccs, 75)) +'\n')
    sys.stdout.write('\n')
    sys.stdout.write('Final-Frame Accuracy: ' + str(np.mean(finAccs)) + ' +/- ' + str(np.std(finAccs)) +'\n')
    sys.stdout.write('25%, 50%, 75% Quartile Final-Frame Accuracy: ' + str(np.percentile(finAccs, 25))
                     + ', ' + str(np.median(finAccs))
                     +  ', ' + str(np.percentile(finAccs, 75)) +'\n')

def nameModelFile(prefix, useGRU, unlabeledModelFile, 
                  numRNNLayers, numRNNNodes, 
                  numDeepLayers, numDeepNodes,
                  numULOut, l2Reg, numUlOut):
    modelFile = '-'.join(['GRU' if useGRU else 'LSTM', 
                          'L', str(numRNNLayers), str(numDeepLayers),
                          'N', str(numRNNNodes), str(numDeepNodes), str(numULOut),
                          'l2', str(l2Reg),
                          unlabeledModelFile])
    if len(prefix) > 0:
        modelFile = prefix + '-' + modelFile
    return modelFile

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
    comprehensiveGestLOOEvaluation(directory = 'UnlabeledDynamic/UnlabeledDynamic', 
                                    modelType='birnn', numEpochs = 500, 
                                    l2Reg = 0.001, learningRate = 0.001, 
                                    useGRU=True, trainMode = 'continue',
                                    gaussianNoise = 0, dropoutH = 0.1,
                                    ULl2Reg = 0.01, ULCenter = True,
                                    ULMasking = True, ULPoolingMode = 'max',
                                    numULOut = 100, numULDeepNodes = 100,
                                    numSpecialNodes = 100, numSpecialOut = 100,
                                    prependMean = False, batchSize = 64)