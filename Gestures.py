# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:15:12 2016

@author: Drew
"""
import csv
import numpy as np
import time

from theano import *
import theano.tensor as T
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
import six.moves.cPickle as pickle
import os
from queryUser import queryUser
import sys
from KerasSupplementary import accuracy, balancedAccuracy, weightedAccuracy

def defaultDirectory():
    return '.'

def loadData(gestureFile = '\\all.left.csv', directory = defaultDirectory(), prune = True):
    gestureFile = directory+gestureFile
    sequenceList = []
    classList = []
    maxSequenceLength = 0
    maxNumFeatures = 0
    with open(gestureFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        sequence = None
        currentClass = 0 #the ID of the class that we are currently loading
        for row in reader:
            if len(row) == 1:
                continue
            if row[0] == 'Start':
                # conclude previous sequence
                if sequence is not None:
                    if sequence.shape[0] > maxSequenceLength:
                        maxSequenceLength = sequence.shape[0]
                    sequenceList.append(sequence)
                #initiate next sequence
                #class IDs are 1-indexed, so make them 0 indexed
                currentClass = int(row[1])-1
                sequence = None
            else:
                # add a frame to the current sequence
                frame = []
                for token in row:
                    frame = frame + [float(token)]
                frame = np.asarray(frame)
                numFeatures = frame.shape[0]
                if numFeatures > maxNumFeatures:
                    #pad previous frames in this sequence with zeros (so that vstack won't crash in a few lines)
                    maxNumFeatures = numFeatures
                    if sequence is not None:
                        sequence = np.hstack((sequence, np.zeros((sequence.shape[0], maxNumFeatures-sequence.shape[1]))))
                elif numFeatures < maxNumFeatures:
                    frame = np.hstack((frame, np.zeros(maxNumFeatures-frame.shape[0])))
                if sequence is None:
                    sequence = frame.reshape((-1, frame.shape[0]))
                    classList = classList + [np.asarray([currentClass])]
                else:
                    sequence = np.vstack((sequence, frame))
                    classList[-1] = np.concatenate((classList[-1], [currentClass]))
        #add last sequence
        if sequence is not None:
            if sequence.shape[0] > maxSequenceLength:
                maxSequenceLength = sequence.shape[0]
            sequenceList.append(sequence)
    
    #remove sequences that are too long
    if prune:
        lengths = np.array([targets.shape[0] for targets in classList])
        upperBound = np.mean(lengths) + 2*np.std(lengths)
        validIndices = (lengths <= upperBound)
        lengths = lengths[validIndices]
        maxSequenceLength = int(np.max(lengths))
        validIndices = validIndices.nonzero()[0].tolist()
        sequenceList = [sequenceList[int(index)] for index in validIndices]
        classList = [classList[int(index)] for index in validIndices]
    
    # pad sequences with dummy data
    def padSequenceData(sequence):
        #pad time
        if sequence.shape[0] < maxSequenceLength:
            sequence = np.concatenate((sequence, np.zeros((maxSequenceLength-sequence.shape[0], sequence.shape[1]))))
        #pad markers
        if sequence.shape[1] < maxNumFeatures:
            sequence = np.concatenate((sequence, np.zeros((sequence.shape[0], maxNumFeatures - sequence.shape[1]))), axis=1)
        return sequence
        
    def padClassData(classID):
        if classID.shape[0] < maxSequenceLength:
            classID = np.concatenate((classID, -np.ones((maxSequenceLength-classID.shape[0],))))
        return classID
        
    sequenceList[:] = [padSequenceData(sequence) for sequence in sequenceList]
    classList[:] = [padClassData(classID) for classID in classList]
    """
    Finish conversion to arrays and switch numSequences and numObservations dimensions
    """
    sequenceList = np.asarray(sequenceList).transpose([1,0,2]) 
    classList = np.asarray(classList).T
    return sequenceList, classList.astype(int)

def loadUserSeparatedData(directory=defaultDirectory(), preExt = '.left',
                          userAbbrv='u', classAbbrv = 'g', classRange = range(1,7)):
    """
    Loads each file separately into its own array.
    """
    userRange = [0, 1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14]
    sequenceLists = []
    classLists = []
    for u in userRange:
        for c in classRange:
            [sequenceList, classList] = loadData(gestureFile='\\'+userAbbrv + str(u) + classAbbrv + str(c) + preExt + '.csv',
                                                        directory=directory, prune = False)
            sequenceLists = sequenceLists + [sequenceList]
            classLists = classLists + [classList]
    return sequenceLists, classLists

def loadUserMergedData(directory=defaultDirectory(), prune = True, preExt = '.left',
                       userAbbrv='u', classAbbrv = 'g', classRange=range(1,7)):
    sequenceLists, classLists = loadUserSeparatedData(directory=directory, preExt=preExt,
                                                      userAbbrv=userAbbrv, classAbbrv=classAbbrv,
                                                      classRange=classRange)
    return mergeData(sequenceLists, classLists, len(classRange), prune)


def mergeData(sequenceLists, classLists, numClasses, prune):
    # merge lists
    maxSequenceLength = max([classList.shape[0] for classList in classLists])
    maxNumFeatures = max([sequenceList.shape[2] for sequenceList in sequenceLists])
    # pad sequences with dummy data
    def padSequenceData(sequenceList):
        if sequenceList.shape[0] < maxSequenceLength:
            sequenceList = np.concatenate((sequenceList, np.zeros((maxSequenceLength-sequenceList.shape[0], 
                                                                   sequenceList.shape[1], 
                                                                   sequenceList.shape[2]))))
        if sequenceList.shape[2] < maxNumFeatures:
            sequenceList = np.concatenate((sequenceList, np.zeros((sequenceList.shape[0], 
                                                                   sequenceList.shape[1], 
                                                                   maxNumFeatures-sequenceList.shape[2]))), axis=2)
        return sequenceList
        
    def padClassData(classList):
        if classList.shape[0] < maxSequenceLength:
            classList = np.concatenate((classList, -np.ones((maxSequenceLength-classList.shape[0], 
                                                             classList.shape[1]))))
        return classList
        
    sequenceLists[:] = [padSequenceData(sequenceList) for sequenceList in sequenceLists]
    classLists[:] = [padClassData(classList) for classList in classLists]
    classPartitions = [sequenceList.shape[1] for sequenceList in sequenceLists]
    classPartitions = np.concatenate(([0], np.cumsum(classPartitions)))
    userPartitions = classPartitions[0::numClasses]
    sequenceLists = np.concatenate(sequenceLists, axis=1)
    classLists = np.concatenate(classLists, axis=1)
    
    if prune:
        lengths = (classLists >= 0).sum(axis=0)
        upperBound = np.mean(lengths) + 2*np.std(lengths)
        validIndices = (lengths <= upperBound)
        invalidIndices = (lengths > upperBound).nonzero()[0].tolist()
        lengths = lengths[validIndices]
        maxSequenceLength = int(np.max(lengths))
        validIndices = validIndices.nonzero()[0].tolist()
        sequenceLists = [sequenceLists[0:maxSequenceLength, int(index), :] for index in validIndices]
        classLists = [classLists[0:maxSequenceLength, int(index)] for index in validIndices]
        sequenceLists = np.asarray(sequenceLists).transpose((1,0,2))
        classLists = np.asarray(classLists).T
        for i in reversed(invalidIndices):
            userPartitions = [p-1 if p > i else p for p in userPartitions]
            classPartitions = [p-1 if p > i else p for p in classPartitions]
    #classPartitions = np.array(classPartitions).reshape((-1, numClasses))
        
    return sequenceLists, classLists.astype(int), userPartitions, classPartitions
    

def normalizePercentages(trainPer, valPer, testPer, totalPer):
    if totalPer <= 0:
        totalPer = 1
    totalPer = min(totalPer, 1) #constrain to (0,1]
    total = float(trainPer + valPer + testPer)
    [trainPer, valPer, testPer] = [per / total for per in [trainPer, valPer, testPer]]
    return trainPer, valPer, testPer, totalPer
    

def partitionData(classes, trainPer, valPer, testPer, totalPer):
    # Select a totalPer percentage of the dataset that keeps classes with the
    # provided balance of representation.
    # Assumes the data has already been randomly permuted.

    #normalize percentages
    trainPer, valPer, testPer, totalPer = normalizePercentages(trainPer, valPer, testPer, totalPer)
    sortedClasses, classIndices = (list(t) for t in zip(*sorted(zip(classes[0,:].tolist(), range(classes.shape[1])))))
    _, startIndices, counts = np.unique(sortedClasses, return_index=True, return_counts=True)
    
    # select balanced totalPer percentage
    counts = [int(count*totalPer) for count in counts]    
    
    def partitionRange(rang, trainPer, valPer, testPer):
        trainRange = range(0, int(np.round(trainPer*len(rang))))
        if len(trainRange) > 0:
            valRange = range(trainRange[-1]+1, trainRange[-1]+1+int(np.round(valPer*len(rang))))
        else:
            valRange = range(0, int(np.round(valPer*len(rang))))
        if len(valRange) > 0:
            testRange = range(valRange[-1]+1, len(rang))
        else:
            testRange = range(0, len(rang))
        trainRange = [rang[i] for i in trainRange]
        valRange = [rang[i] for i in valRange]
        testRange = [rang[i] for i in testRange]
        return trainRange, valRange, testRange
    
    ranges = [partitionRange(classIndices[index:(index+count)], trainPer, valPer, testPer) 
              for index, count in zip(startIndices, counts)]
    
    trainRange = []
    valRange = []
    testRange = []
    for r in ranges:
        trainRange += r[0]
        valRange += r[1]
        testRange += r[2]
    
    return trainRange, valRange, testRange
    
    
def loadDataset(directory=defaultDirectory(), delRange=range(0,18),
                trainPer=0.6, valPer=0.25, 
                testPer=0.15, totalPer=1, 
                preExt='.left', prune=True,
                userAbbrv='u', classAbbrv = 'g',
                classRange = range(1,7), LOUO = False,
                trainAbs = None, valAbs = None, testAbs = None):
    [sequences, classes, 
     userPartitions, classPartitions] = loadUserMergedData(directory = directory, prune = prune, preExt=preExt,
                                                           userAbbrv=userAbbrv, classAbbrv=classAbbrv,
                                                           classRange=classRange)
    # prune global coordinate data?
    if delRange is not None:
        sequences = np.delete(sequences, delRange, axis=2)
    [numObservations, numSequences, numFeatures] = sequences.shape
    numClasses = int(np.max(classes)+1)
    if LOUO:
        returnStructs = []
        classRanges = []
        maxRange = 0
        for cid in range(0, len(classPartitions)-1):
            classRange = np.array(range(classPartitions[cid], classPartitions[cid+1]))
            #go ahead and permute each class separately
            perm = np.random.permutation(len(classRange))
            classRange = classRange[perm]
            maxRange = max([maxRange,len(classRange)])
            classRanges += [classRange]
        #make sure parameters make sense
        def noneSum(a,b):
            if a is None and b is None:
                return None
            elif a is None:
                return b
            elif b is None:
                return a
            else:
                return a+b
        if ((trainAbs is None and valAbs is not None) 
            or (trainAbs is not None and valAbs is None)):
            raise ValueError('Training and validation sets must be selected in the same manner, either proportionally or absolutely.')
        [trainPer, valPer, _, totalPer] = normalizePercentages(trainPer, valPer, 0, totalPer)
        [_, _, testPer, totalPer] = normalizePercentages(0, 0, testPer, totalPer)
        maxRange = max([maxRange, noneSum(trainAbs, valAbs), testAbs])
        #in addition, balance each class representation via bootstrapping
        def bootstrap(arr, numSamples):
            sample = np.random.choice(arr, numSamples)
            return np.concatenate((arr, sample))
        classRanges = [bootstrap(classRange, maxRange-classRange.shape[0]) for classRange in classRanges]
        for u in range(0, len(userPartitions)-1):
            testRange = [classRanges[r] for r in range(u*numClasses, (u+1)*numClasses)]
            trainValRange = [classRanges[r] for r in (range(0, u*numClasses)+range((u+1)*numClasses, len(classRanges)))]
            if testAbs is not None:
                testRange = [rang[0:testAbs] for rang in testRange]
            else:
                testRange = [rang[0:(rang.shape[0]*totalPer)] for rang in testRange]
            if trainAbs is not None and valAbs is not None: 
                trainRange = [rang[0:trainAbs] for rang in trainValRange]
                valRange = [rang[trainAbs:(trainAbs+valAbs)] for rang in trainValRange]
            else:
                trainRange = [rang[0:int(rang.shape[0]*trainPer*totalPer)] for rang in trainValRange]
                valRange = [rang[int(rang.shape[0]*trainPer*totalPer):int(rang.shape[0]*totalPer)] for rang in trainValRange]
            trainRange = np.concatenate(trainRange).tolist()
            valRange = np.concatenate(valRange).tolist()
            testRange = np.concatenate(testRange).tolist()
            #randomly permute data to mix the classes into eachother
            def permuteList(lis):
                perm = np.random.permutation(len(lis)).tolist()
                lis = [lis[p] for p in perm]
                return lis
            trainRange = permuteList(trainRange)
            valRange = permuteList(valRange)
            testRange = permuteList(testRange)
            returnStruct = [(sequences, classes, trainRange, 
                            valRange, testRange, numClasses,
                            numObservations, numSequences, numFeatures)]
            returnStructs += returnStruct          
        return returnStructs
    else:
        #randomly permute data
        perm = np.random.permutation(numSequences)
        sequences[:,:,:] = sequences[:, perm, :]
        classes[:, :] = classes[:, perm]
        #reduce to indicated percentage of dataset
        #separate into training, validation, and testing partitions
        trainRange, valRange, testRange = partitionData(classes, trainPer, 
                                                        valPer, testPer, totalPer)
        returnStruct = (sequences, classes, trainRange, 
                        valRange, testRange, numClasses,
                        numObservations, numSequences, numFeatures)
        return returnStruct
    

def comprehensiveEvaluation(directory = defaultDirectory(),  
                                pruneGlobal = True, numLayers = 2, 
                                numNodesPerLayer = 200, randSeed = 1,
                                trainPer = .6, valPer = .25, testPer = .15,
                                totalPer = 1, batchSize = 64,
                                numEpochs = 1000, learningRate = 0.001, 
                                l2Reg = 0.0001, modelFile = None,
                                useGRU = False,
                                dropoutI = 0.2, dropoutH=0.2, 
                                trainMode = 'continue', randSeed2 = None):
    """
    Train an RNN for gesture recognition on samples taken from each user.
    """
    trainPer, valPer, testPer, totalPer = normalizePercentages(trainPer, valPer, testPer, totalPer)
    if modelFile is None:
        modelFile = nameModelFile('', useGRU, numLayers, numNodesPerLayer, randSeed,
                                  trainPer, valPer, testPer, totalPer, dropoutI, dropoutH, l2Reg)
    
    np.random.seed(randSeed) #control permutation of data
    # prune global coordinate data?
    if pruneGlobal:
        pruneRange = range(0, 18)
    else:
        pruneRange = None
    
    struct = loadDataset(directory, pruneRange, trainPer, valPer, 
                         testPer, totalPer, '.left', True)
    
    if randSeed2 is not None: #control randomization of training
        np.random.seed(randSeed2)
    trainGestureRNN(numLayers=numLayers, numNodesPerLayer=numNodesPerLayer,
                    useGRU=useGRU, batchSize=batchSize, 
                    numEpochs = numEpochs, learningRate=learningRate,
                    l1Reg=0, l2Reg = l2Reg, dropoutI=dropoutI, dropoutH=dropoutH,
                    sequences = struct[0], classes = struct[1],
                    trainRange = struct[2], valRange = struct[3],
                    testRange = struct[4], numClasses = struct[5],
                    numObservations = struct[6], numSequences = struct[7],
                    numFeatures = struct[8],
                    modelFile=modelFile, 
                    trainMode=trainMode,
                    callbacks = [EarlyStopping(patience=20)])

def comprehensiveLOOEvaluation(directory=defaultDirectory(),   
                               pruneGlobal = True, numLayers = 2, 
                               numNodesPerLayer = 200, randSeed = 1,
                               trainPer = .6, valPer = .25, testPer = 0.15,
                               totalPer = 1, batchSize = 64,
                               numEpochs = 1000, learningRate = 0.001, 
                               l2Reg = 0.0001, modelFilePrefix = '',
                               useGRU = False,
                               dropoutI = 0.2, dropoutH = 0.2, trainMode = 'continue',
                               randSeed2 = None, center = False, prependMean = False):
    """
    Train RNNs for a leave-one-user-out evaluation.
    """
    
    trainModes = ['continue', 'overwrite', 'continue-each']
    
    if trainMode.lower() not in trainModes:
        raise ValueError("Parameter 'trainMode' must be either 'continue', 'overwrite', or 'continue-each'.")
    
    np.random.seed(randSeed) #control permutation of data
    # prune global coordinate data?
    if pruneGlobal:
        pruneRange = range(0, 18)
    else:
        pruneRange = None
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
    outDirectory = nameModelFile('', useGRU, numLayers, numNodesPerLayer, randSeed,
                                 trainPer, valPer, testPer, totalPer, dropoutI, dropoutH, l2Reg,
                                 center, prependMean)
    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)
    if randSeed2 is not None: #control randomization of training (for Keras at least)
        np.random.seed(randSeed2)
    for struct in structs:
        modelFile = modelFilePrefix + 'LOU-' + str(u)
        modelFile = nameModelFile(modelFile, useGRU, numLayers, numNodesPerLayer, randSeed,
                                  trainPer, valPer, testPer, totalPer, dropoutI, dropoutH, l2Reg,
                                  center, prependMean)
        u += 1
        if (os.path.isfile(outDirectory + '\\' + 'Keras' + modelFile + '.json') 
            and os.path.isfile(outDirectory + '\\' +  'Keras' + modelFile + '_Weights.h5')):
            #if we have already trained for leaving out this user
            if trainMode == 'continue': #continue until each user has a model
                trainMode2 = 'skip' 
            elif trainMode == 'continue-each': # continue training previous models
                trainMode2 = 'continue'
            else:
                trainMode2 = 'overwrite'
        else:
            trainMode2 = trainMode

        if center:
            """
            Center the labeled markers on their mean. 
            """
            from Postures import centerData
            struct = list(struct)
            labeledMarkerData = struct[0][:,:,18:].reshape((-1, 11, 3))
            labeledMarkerData = centerData(labeledMarkerData, True, prependMean).reshape((struct[0].shape[0], struct[0].shape[1], -1))
            struct[0] = np.concatenate([struct[0][:,:,0:18], labeledMarkerData], axis = 2)
            if prependMean:
                struct[8] += 3


        cmEpoch, loss, acc, balAcc, finAcc = trainGestureRNN(numLayers=numLayers, numNodesPerLayer=numNodesPerLayer,
                                                             useGRU=useGRU, batchSize=batchSize, 
                                                             numEpochs = numEpochs, learningRate=learningRate,
                                                             l1Reg=0, l2Reg = l2Reg, dropoutI=dropoutI, dropoutH=dropoutH,
                                                             sequences = struct[0], classes = struct[1],
                                                             trainRange = struct[2], valRange = struct[3],
                                                             testRange = struct[4], numClasses = struct[5],
                                                             numObservations = struct[6], numSequences = struct[7],
                                                             numFeatures = struct[8],
                                                              modelFile=modelFile, 
                                                             outDirectory=outDirectory, trainMode=trainMode2,
                                                             callbacks = [EarlyStopping(patience=20)])
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
    sys.stdout.write('Leave One User Out Evaluation\nTest Results for ' + str(numLayers) + '-Layer, ' 
                     + str(numNodesPerLayer) + ' Nodes-Per-Layer ' + ('GRU' if useGRU else 'LSTM') + ' Networks\n'
                     + 'Trained with ' + ("%0.2f" % (dropoutI*100)) + '% Input Dropout, '
                     + ("%0.2f" % (dropoutH*100)) + '% Hidden Dropout, and ' + str(l2Reg) + ' L2 Regularization\n'
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
    
        
def trainGestureRNN(numLayers, numNodesPerLayer, useGRU, batchSize,
                    numEpochs, learningRate, l1Reg, l2Reg, dropoutI, dropoutH,
                    sequences, classes, trainRange, valRange, testRange,
                    numClasses, numObservations, numSequences, numFeatures,
                    modelFile, callbacks = None, 
                    outDirectory = '', trainMode = 'continue'):
    """
    Returns True if training was completed, False if interrupted.
    """
    trainModes = ['continue', 'overwrite', 'skip']
    
    if trainMode.lower() not in trainModes:
        raise ValueError("Parameter 'trainMode' must be one of 'continue', 'overwrite', or 'skip'")
    
    if dropoutI < 0 or dropoutH < 0 or l2Reg < 0 or l1Reg < 0:
        raise ValueError('Regularization parameters must be non-negative.')
    
    if outDirectory is not None and outDirectory != '':
        outDirectory = outDirectory + '\\'
    else:
        outDirectory = ''
    # initialize, compile, and train model
    #finish preparing data
    #class labels must be made into binary arrays
    binaryClasses = np.zeros((numObservations, numSequences, numClasses))
    # tell cost function which timesteps to ignore
    sampleWeights = np.ones((numObservations, numSequences))
    #eh...just use for loops
    for i in range(numObservations):
        for j in range(numSequences):
            if classes[i,j] >= 0:
                binaryClasses[i,j, classes[i,j]] = 1
            else:
                sampleWeights[i,j] = 0
    sequences = sequences.transpose((1,0,2))
    binaryClasses = binaryClasses.transpose((1,0,2))
    sampleWeights = sampleWeights.T
        
    trainData = [sequences[trainRange,:,:], binaryClasses[trainRange,:,:], sampleWeights[trainRange, :]]
    valData = [sequences[valRange,:,:], binaryClasses[valRange,:,:], sampleWeights[valRange, :]]
    testData = [sequences[testRange, :, :], binaryClasses[testRange, :, :], sampleWeights[testRange, :]]
        
    modelFile = outDirectory + 'Keras'+modelFile
    weightsFile = modelFile+'_Weights'
    completedEpochs = 0
    if (trainMode == 'overwrite') or (not os.path.isfile(modelFile+'.json') or not os.path.isfile(weightsFile+'.h5')):
        model = Sequential()
        #add masking layer to indicate dummy timesteps
        model.add(Masking(0, input_shape=(numObservations, numFeatures)))
        if dropoutI:
            model.add(Dropout(dropoutI))
        for i in range(numLayers):
            if useGRU:
                model.add(GRU(output_dim=numNodesPerLayer, return_sequences=True,
                                W_regularizer=l2(l2Reg)))
            else:
                model.add(LSTM(output_dim=numNodesPerLayer, return_sequences=True,
                                W_regularizer=l2(l2Reg)))
            if dropoutH:
                model.add(Dropout(dropoutH))
        model.add(TimeDistributed(Dense(output_dim=numClasses, activation='softmax', 
                                        W_regularizer = l2(l2Reg))))
    else:
        model = model_from_json(open(modelFile+'.json', 'rb').read())
        model.load_weights(weightsFile+'.h5')
        
    #compile model and training objective function
    sgd = SGD(lr=learningRate)
    rms = RMSprop(lr=learningRate)
    adagrad = Adagrad(lr=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer=rms,
                    sample_weight_mode='temporal', metrics=['accuracy'])
    checkp = [ModelCheckpoint(weightsFile + '.h5', save_best_only = True)]
    if callbacks is None:
        callbacks = checkp
    else:
        callbacks += checkp
    try:
        if trainMode != 'skip':
            completedEpochs = model.fit(x=trainData[0], y=trainData[1], sample_weight=trainData[2],
                                        validation_data = valData, batch_size = batchSize, 
                                        nb_epoch = numEpochs, callbacks = callbacks,
                                        verbose = 2)
            completedEpochs = len(completedEpochs.history['loss'])
    except KeyboardInterrupt:
        if(not queryUser('Training interrupted. Compute test statistics?')):
            return 0, float('nan'), float('nan'), float('nan') 
    #retrieve the best weights based upon validation set loss
    if os.path.isfile(weightsFile+'.h5'):
        model.load_weights(weightsFile+'.h5')
    scores = model.test_on_batch(x=testData[0], y=testData[1], sample_weight=testData[2])
    predictedClasses = model.predict_classes(x=testData[0])
    scores[1] = accuracy(classes[:, testRange].T, predictedClasses)
    scores.append(balancedAccuracy(classes[:, testRange].T, predictedClasses))
    scores.append(weightedAccuracy(classes[:, testRange].T, predictedClasses, forgetFactor=0))
    print("Test loss of %.5f\nFrame-wise accuracy of %.5f\nSequence-wise accuracy of %.5f\nFinal frame accuracy of %0.5f" % (scores[0], scores[1], scores[2], scores[3]))
    if trainMode != 'skip':
        modelString = model.to_json()
        open(modelFile + '.json', 'wb').write(modelString)
        model.save_weights(weightsFile + '.h5', overwrite=True)
        print('Model and weights saved to %s and %s.' % (modelFile+'.json', weightsFile+'.h5'))
    return completedEpochs, scores[0], scores[1], scores[2], scores[3]

def nameModelFile(prefix, useGRU, numLayers, numNodesPerLayer,
                  randSeed, trainPer, valPer, testPer, totalPer,
                  dropoutI, dropoutH, l2Reg, 
                  center = False, prependMean = False):
    
    modelFile = (prefix + ('GRU' if useGRU else 'LSTM') +'-L' + str(numLayers) 
                    +'-N' + str(numNodesPerLayer)+'-S'+str(randSeed) + '-TS-' 
                    + str(trainPer) + '-' + str(valPer) + '-' +  str(testPer) 
                    + '-' +  str(totalPer)+ '-l2-' + str(l2Reg) 
                    + (('-D-' + str(dropoutI) +'-'+ str(dropoutH)) if (dropoutI or dropoutH) else '')
                    + (('-C' + ('P' if prependMean else '')) if center else ''))
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
    comprehensiveLOOEvaluation(directory='FilteredAndLabeledGestureData/LabeledDynamic', 
                            numEpochs = 500, batchSize = 128, numNodesPerLayer = 200,
                            numLayers = 2, learningRate = .001, totalPer = 1, 
                            dropoutI = 0, dropoutH = 0.1, useGRU = True, l2Reg = 0.001,
                            center = False, prependMean = False,
                            trainMode = 'continue')