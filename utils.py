#!/usr/bin/env python
# coding: utf-8


import numpy as np


from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import os
import hickle as hkl 
import tensorflow as tf
import seaborn as sns
import logging
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

class dataHolder:
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    activityLabels = []
    clientCount = None

    
def returnClientByDataset(dataSetName):
    if(dataSetName=='UCI' or dataSetName ==  'UCI_ORIGINAL'):
        return 5
    elif(dataSetName == "RealWorld" ):
        return 15
    elif(dataSetName == "MotionSense"):
        return 24 
    elif(dataSetName == 'SHL'):
        return 9
    elif(dataSetName == "HHAR"):
        return 51
    else:
        raise ValueError('Unknown dataset')
    
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None)
    return dataframe.values


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    loaded = np.dstack(loaded)
    return loaded


def load_dataset(group,mainDir,prefix=''):
    filepath = mainDir + 'datasetStandardized/'+prefix + '/' + group + '/'
    filenames = list()
    filenames += ['AccX'+prefix+'.csv', 'AccY' +
                prefix+'.csv', 'AccZ'+prefix+'.csv']
    filenames += ['GyroX'+prefix+'.csv', 'GyroY' +
                prefix+'.csv', 'GyroZ'+prefix+'.csv']
    X = load_group(filenames, filepath)
    y = load_file(mainDir + 'datasetStandardized/'+prefix +
                '/' + group + '/Label'+prefix+'.csv')
    return X, y

def projectTSNE(fileName,filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels):
    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        palette=sns.color_palette(n_colors = len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)]) 
        

    plt.tick_params(
    axis='both',         
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,        
    labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath+fileName+".svg", bbox_inches="tight", format="svg")
    plt.show()

def projectTSNEWithPosition(dataSetName,fileName,filepath,ACTIVITY_LABEL,labels_argmax,orientationsNames,clientOrientationTest,tsne_projections,unique_labels):
    classData = [ACTIVITY_LABEL[i] for i in labels_argmax]
    orientationData = [orientationsNames[i] for i in np.hstack((clientOrientationTest))]
    if(dataSetName == 'RealWorld'):
        orientationName = 'Position'
    else:
        orientationName = 'Device'
    pandaData = {'col1': tsne_projections[:,0], 'col2': tsne_projections[:,1],'Classes':classData, orientationName :orientationData}
    pandaDataFrame = pd.DataFrame(data=pandaData)

    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    sns.scatterplot(data=pandaDataFrame, x="col1", y="col2", hue="Classes", style=orientationName,
                    palette=sns.color_palette(n_colors = len(unique_labels)),
                    s=50, alpha=1.0,rasterized=True,)
    plt.tick_params(
    axis='both',          
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,       
    labelbottom=False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath+fileName+".png", bbox_inches="tight")
    plt.show()
    
def create_segments_and_labels_Mobiact(df, time_steps, step, label_name = "LabelsEncoded", n_features= 6):
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        acc_x = df['acc_x'].values[i: i + time_steps]
        acc_y = df['acc_y'].values[i: i + time_steps]
        acc_z = df['acc_z'].values[i: i + time_steps]

        gyro_x = df['gyro_x'].values[i: i + time_steps]
        gyro_y = df['gyro_y'].values[i: i + time_steps]
        gyro_z = df['gyro_z'].values[i: i + time_steps]

    

        # Retrieve the most often used label in this segment
        label = scipy.stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)

    return reshaped_segments, labels
    
    
def loadDataset(dataSetName, clientCount, dataConfig, randomSeed, mainDir, StratifiedSplit = True):
# loading datasets
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    
    
    if(dataSetName == "UCI"):

        centralTrainData = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/trainX.hkl')
        centralTestData = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/testX.hkl')
        centralTrainLabel = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/trainY.hkl')
        centralTestLabel = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/testY.hkl')

        
    elif(dataSetName == "SHL"):
        clientData = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/clientsData.hkl')
        clientLabel = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/clientsLabel.hkl')
        clientCount = clientData.shape[0]
        
        for i in range(0,clientCount):
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            skf.get_n_splits(clientData[i], clientLabel[i])
            trainIndex = []
            testIndex = []
            for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
        #             let indices at index 4 be used for test
                if(enu_index != 2):
                    trainIndex.append(test_index)
                else:
                    testIndex = test_index
            trainIndex = np.hstack((trainIndex))
            clientDataTrain.append(clientData[i][trainIndex])
            clientLabelTrain.append(clientLabel[i][trainIndex])
            clientDataTest.append(clientData[i][testIndex])
            clientLabelTest.append(clientLabel[i][testIndex])
        clientDataTrain = np.asarray(clientDataTrain,dtype  = object)
        clientDataTest = np.asarray(clientDataTest,dtype  = object)

        clientLabelTrain = np.asarray(clientLabelTrain,dtype  = object)
        clientLabelTest = np.asarray(clientLabelTest,dtype  = object)

        centralTrainData = np.vstack((clientDataTrain))
        centralTrainLabel = np.hstack((clientLabelTrain))

        centralTestData = np.vstack((clientDataTest))
        centralTestLabel = np.hstack((clientLabelTest))
        
    elif(dataSetName == "RealWorld"):
        orientationsNames = ['chest','forearm','head','shin','thigh','upperarm','waist']
        clientDataTrain = {new_list: [] for new_list in range(clientCount)}
        clientLabelTrain = {new_list: [] for new_list in range(clientCount)}
        clientDataTest = {new_list: [] for new_list in range(clientCount)}
        clientLabelTest = {new_list: [] for new_list in range(clientCount)}
        
        clientOrientationData = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/clientsData.hkl')
        clientOrientationLabel = hkl.load(mainDir + 'datasetStandardized/'+str(dataSetName)+'/clientsLabel.hkl')
        

        clientOrientationTest = {new_list: [] for new_list in range(clientCount)}
        clientOrientationTrain = {new_list: [] for new_list in range(clientCount)}

        orientationIndex = 0
        for clientData,clientLabel in zip(clientOrientationData,clientOrientationLabel):
            for i in range(0,clientCount):
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                skf.get_n_splits(clientData[i], clientLabel[i])
                trainIndex = []
                testIndex = []
                for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
        #             let indices at index 2 be used for test
                    if(enu_index != 2):
                        trainIndex.append(test_index)
                    else:
                        testIndex = test_index

                trainIndex = np.hstack((trainIndex))
                clientDataTrain[i].append(clientData[i][trainIndex])
                clientLabelTrain[i].append(clientLabel[i][trainIndex])
                clientDataTest[i].append(clientData[i][testIndex])
                clientLabelTest[i].append(clientLabel[i][testIndex])

                clientOrientationTest[i].append(np.full((len(testIndex)),orientationIndex))
                clientOrientationTrain[i].append(np.full((len(trainIndex)),orientationIndex))

            orientationIndex += 1
                
        for i in range(0,clientCount):
            clientDataTrain[i] = np.vstack((clientDataTrain[i]))
            clientDataTest[i] = np.vstack((clientDataTest[i]))
            clientLabelTrain[i] = np.hstack((clientLabelTrain[i]))
            clientLabelTest[i] = np.hstack((clientLabelTest[i]))
            clientOrientationTest[i] = np.hstack((clientOrientationTest[i]))
            clientOrientationTrain[i] = np.hstack((clientOrientationTrain[i]))

        clientOrientationTrain = np.asarray(list(clientOrientationTrain.values()),dtype  = object)
        clientOrientationTest = np.asarray(list(clientOrientationTest.values()),dtype  = object)
        
        
        clientDataTrain = np.asarray(list(clientDataTrain.values()),dtype  = object)
        clientDataTest = np.asarray(list(clientDataTest.values()),dtype  = object)
        
        clientLabelTrain = np.asarray(list(clientLabelTrain.values()),dtype  = object)
        clientLabelTest = np.asarray(list(clientLabelTest.values()),dtype  = object)

        centralTrainData = np.vstack((clientDataTrain))
        centralTrainLabel = np.hstack((clientLabelTrain))

        centralTestData = np.vstack((clientDataTest))
        centralTestLabel = np.hstack((clientLabelTest))

    else:
        clientData = []
        clientLabel = []

        for i in range(0,clientCount):
            clientData.append(hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/UserData'+str(i)+'.hkl'))
            clientLabel.append(hkl.load(mainDir + 'datasetStandardized/'+dataSetName+'/UserLabel'+str(i)+'.hkl'))
            
        if(dataSetName == "HHAR"):
            orientations = hkl.load(mainDir + 'datasetStandardized/HHAR/deviceIndex.hkl')
            orientationsNames = ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']

        for i in range (0,clientCount):
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            skf.get_n_splits(clientData[i], clientLabel[i])
            partitionedData = list()
            partitionedLabel = list()    
            dataIndex = []
            trainIndex = []
            testIndex = []
            for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
                if(enu_index != 2):
                    trainIndex.append(test_index)
                else:
                    testIndex = test_index
            trainIndex = np.hstack((trainIndex))
            clientDataTrain.append(clientData[i][trainIndex])
            clientLabelTrain.append(clientLabel[i][trainIndex])
            clientDataTest.append(clientData[i][testIndex])
            clientLabelTest.append(clientLabel[i][testIndex])
            clientOrientationTrain.append(trainIndex)
            clientOrientationTest.append(testIndex) 
            
        if(dataSetName == "HHAR"):        
            for i in range(0,clientCount):
                clientOrientationTest[i] = orientations[i][clientOrientationTest[i]]
                clientOrientationTrain[i] = orientations[i][clientOrientationTrain[i]]

            
            
        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))


    dataReturn = dataHolder
    dataReturn.clientDataTrain = clientDataTrain
    dataReturn.clientLabelTrain = clientLabelTrain
    dataReturn.clientDataTest = clientDataTest
    dataReturn.clientLabelTest = clientLabelTest
    dataReturn.centralTrainData = centralTrainData
    dataReturn.centralTrainLabel = centralTrainLabel
    dataReturn.centralTestData = centralTestData
    dataReturn.centralTestLabel = centralTestLabel
    dataReturn.clientOrientationTrain = clientOrientationTrain
    dataReturn.clientOrientationTest = clientOrientationTest
    dataReturn.orientationsNames = orientationsNames
    return dataReturn


def plot_learningCurve(history, epochs, filepath):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'],markevery=[np.argmax(history.history['val_accuracy'])], ls="", marker="o",color="orange")
    plt.plot(epoch_range, history.history['accuracy'],markevery=[np.argmax(history.history['accuracy'])], ls="", marker="o",color="blue")

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.savefig(filepath+"LearningAccuracy.svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()
    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.plot(epoch_range, history.history['loss'],markevery=[np.argmin(history.history['loss'])], ls="", marker="o",color="blue")
    plt.plot(epoch_range, history.history['val_loss'],markevery=[np.argmin(history.history['val_loss'])], ls="", marker="o",color="orange")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(filepath+"ModelLoss.svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()


def roundNumber(toRoundNb):
    return round(toRoundNb, 4) * 100

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    """
    Create an intermediate model from base mode, which outputs embeddings of the intermediate layer

    Parameters:
        base_model
            the base model from which the intermediate model is built
        
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

    Returns:
        model (tf.keras.Model)
    """

    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model