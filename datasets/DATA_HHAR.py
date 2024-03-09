#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment if running on googlecolab 
# !pip install hickle
# from google.colab import drive
# drive.mount('/content/drive/')
# %cd drive/MyDrive/PerCom2021-FL-master/


# In[ ]:


import hickle as hkl 
import numpy as np
import os
import pandas as pd
from subprocess import call
import requests 
np.random.seed(0)
import urllib.request
import zipfile
from scipy import signal
import urllib.request


# In[ ]:





# In[ ]:


# functions for loading and downloading the dataset

# load a single file as a numpy array
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded
 
# load a dataset group, such as train or test
def load_dataset(group, prefix='',position=''):
	filepath = prefix + '/' + group + '/' + position
	filenames = list()
	# body acceleration
	filenames += ['Acc_x.txt', 'Acc_y.txt', 'Acc_z.txt']
	# body gyroscope
	filenames += ['Gyr_x.txt', 'Gyr_y.txt', 'Gyr_z.txt']
	# load input data
	x = np.asarray(load_group(filenames, filepath))
	# load class output
	y =  processLabel(load_file(filepath+'/Label.txt'))
	return x, y

# download function for datasets
def download_url(url, save_path, chunk_size=8192):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# In[ ]:


fileName = ["Activity recognition exp"]
links = ["http://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"]


# In[ ]:


# download and unzipping dataset/download
os.makedirs('dataset/download',exist_ok=True)
os.makedirs('dataset/extracted',exist_ok=True)

for i in range(len(fileName)):
    data_directory = os.path.abspath("dataset/download/"+str(fileName[i])+".zip")
    if not os.path.exists(data_directory):
        print("downloading "+str(fileName[i]))            
        download_url(links[i],data_directory)
        print("download done")
    else:
        print(str(fileName[i]) + " already downloaded")
        
    extract_directory = os.path.abspath("dataset/extracted/"+str(fileName[i]))
    if not os.path.exists(extract_directory):
        print("extracting data...")
        with zipfile.ZipFile(data_directory, 'r') as zip_ref:
            zip_ref.extractall(os.path.abspath("dataset/extracted/"))
        print("data extracted")
    else:
        print("data already extracted")


# In[ ]:


def findRanges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def unionRange(a):
    b = []
    for begin,end in sorted(a):
        if b and b[-1][1] >= begin - 1:
            b[-1][1] = max(b[-1][1], end + 1)
        else:
            b.append([begin, end + 1])
    return b


# In[ ]:


def processLabel(labels):
    uniqueCount = np.unique(labels,return_counts=True)
    if(len(uniqueCount[0]) > 1):
        return uniqueCount[0][np.argmax(uniqueCount[1])]
    else:
        return uniqueCount[0][0]


# In[ ]:


def downSampleLowPass(motionData,factor):
    accX = signal.decimate(motionData[:,:,0],factor)
    accY = signal.decimate(motionData[:,:,1],factor)
    accZ = signal.decimate(motionData[:,:,2],factor)
    gyroX = signal.decimate(motionData[:,:,3],factor)
    gyroY = signal.decimate(motionData[:,:,4],factor)
    gyroZ = signal.decimate(motionData[:,:,5],factor)
    return np.dstack((accX,accY,accZ,gyroX,gyroY,gyroZ))


# In[ ]:


def segmentData(accData,time_step,step):
#     print(accData.shape)
    step = int(step)
    segmentAccData = []
    for i in range(0, accData.shape[0] - time_step,step):
#         dataSlice = accData[i:i+time_step,:]
#         dataSlice = np.delete(dataSlice,sliceIndex,  0)
#         segmentAccData.append(dataSlice)
#         segmentAccData.append(signal.decimate(accData[i:i+time_step,:],2))
        segmentAccData.append(accData[i:i+time_step,:])


    return np.asarray(segmentAccData)
def segmentLabel(accData,time_step,step):
#     print(accData.shape)
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(processLabel(accData[i:i+time_step]))
    return np.asarray(segmentAccData)


def formatData(data,dim):
    remainders = data.shape[0]%dim
    max_index = data.shape[0] - remainders
    data = data[:max_index,:]
    new = np.reshape(data, (-1, 128,3))
    return new


# In[ ]:


def consecutive(data, treshHoldSplit,stepsize=1):
    splittedData = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    returnResults= [newArray for newArray in splittedData if len(newArray)>=treshHoldSplit]
    return returnResults


# In[ ]:


# def consecutive(data, treshHoldSplit,stepsize=1):
#     splittedData = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
#     returnResults = []
#     for newArray in splittedData:
#         if(len(newArray)!=0):
#             if(newArray[0] >= treshHoldSplit):
#                 returnResults.append((newArray[0]))
#     return returnResults


# In[ ]:


dataDir = "dataset/extracted/Activity recognition exp"


# In[ ]:


def prepareData(dataDir,dataDirectory):
    loadedData = load_file(dataDir+"/"+dataDirectory)[1:]
    dataInstanceCount = loadedData.shape[0]
    returnData = []
    for i in range(dataInstanceCount):
        returnData.append(np.asarray(loadedData[i][0].split(",")))
    return returnData


# In[ ]:


# deviceSamplingRate = [200,200, 200,200,150,150,100,100,100,100,50,50]
# deviceWindowFrame = [512,512,512,512,384,384,256,256,256,256,128,128]
# downSamplingRate = [4,4,4,4,3,3,2,2,2,2,1,1]


# In[ ]:


loadList = ['Phones_accelerometer.csv','Phones_gyroscope.csv','Watch_accelerometer.csv','Watch_gyroscope.csv']
classCounts = ['sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'bike']
deviceCounts = ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']
deviceSamplingRate = [200,200,150,100,100,50]
deviceWindowFrame = [512,512,384,256,256,128]
downSamplingRate = [4,4,3,2,2,1]
subDeviceCounts = ['nexus4_1', 'nexus4_2', 'lgwatch_1', 'lgwatch_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2','gear_1', 'gear_2','samsungold_1', 'samsungold_2']
userCounts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


# In[ ]:


unprocessedAccData = prepareData(dataDir,"Phones_accelerometer.csv")
unprocessedGyroData = prepareData(dataDir,"Phones_gyroscope.csv")


# In[ ]:


watchAccData = prepareData(dataDir,"Watch_accelerometer.csv")
watchGyroData = prepareData(dataDir,"Watch_gyroscope.csv")


# In[ ]:


unprocessedAccData = np.vstack((unprocessedAccData,watchAccData))
unprocessedGyroData = np.vstack((unprocessedGyroData,watchGyroData))


# In[ ]:


allProcessedData = {}
allProcessedLabel = {}
deviceIndex = {}


# In[ ]:


clientCount = len(deviceCounts) * len(userCounts)
deviceIndexes = {new_list: [] for new_list in range(len(deviceCounts))}
indexOffset = 0


# In[ ]:


for clientDeviceIndex, deviceName in enumerate(deviceCounts):
    print("Processsing device "+str(deviceName))
    for clientIDIndex, clientIDName in enumerate(userCounts):
        print("Processsing device:"+str(clientDeviceIndex)+" client "+str(clientIDIndex))

        processedClassData = []
        processedClassLabel = []
        dataIndex = (unprocessedAccData[:,6] == clientIDName) & (unprocessedAccData[:,7] == deviceName)
        userDeviceDataAcc = unprocessedAccData[dataIndex]
        if(len(userDeviceDataAcc) == 0):
            print("No acc data found")
            print("Skipping device :"+str(deviceName) + " Client: "+str(clientIDName))
            indexOffset += 1
            continue
        userDeviceDataGyro = unprocessedGyroData[(unprocessedGyroData[:,6] == clientIDName) & (unprocessedGyroData[:,8] == deviceName)]
        if(len(userDeviceDataGyro) == 0):
            userDeviceDataGyro = unprocessedGyroData[np.where(dataIndex == True)[0]]
            
        for classIndex, className in enumerate(classCounts):
            if(len(userDeviceDataAcc) <= len(userDeviceDataGyro)):
                classData = np.where(userDeviceDataAcc[:,9] == className)[0]
            else:
                classData = np.where(userDeviceDataGyro[:,9] == className)[0]
            segmentedClass = consecutive(classData,deviceWindowFrame[int(clientDeviceIndex/2)])
            for segmentedClassRange in (segmentedClass):
                combinedData = np.dstack((segmentData(userDeviceDataAcc[segmentedClassRange][:,3:6],deviceWindowFrame[clientDeviceIndex],deviceWindowFrame[clientDeviceIndex]/2),segmentData(userDeviceDataGyro[segmentedClassRange][:,3:6],deviceWindowFrame[clientDeviceIndex],deviceWindowFrame[clientDeviceIndex]/2)))
                processedClassData.append(combinedData)
                processedClassLabel.append(np.full(combinedData.shape[0], classIndex, dtype=int))
        deviceCheckIndex = clientDeviceIndex % 2
        tempProcessedData = np.vstack((processedClassData))
        if(clientDeviceIndex < 5):
            tempProcessedData =  downSampleLowPass(np.float32(tempProcessedData),downSamplingRate[clientDeviceIndex])
        dataIndex = (len(userCounts) * clientDeviceIndex) + clientIDIndex - indexOffset
        print("Index is at "+str(dataIndex))
        allProcessedData[dataIndex] = tempProcessedData
        allProcessedLabel[dataIndex] = np.hstack((processedClassLabel))
        deviceIndex[dataIndex] = np.full(allProcessedLabel[dataIndex].shape[0], clientDeviceIndex)
        deviceIndexes[clientDeviceIndex].append(dataIndex)
#             print(str(len(allProcessedData)) + " at "+ str(clientIDName) + " and device " + str(deviceName))
#             print(allProcessedLabel[(clientIDIndex * 6) + clientDataIndex].shape)
                


# In[ ]:


allProcessedData = np.asarray(list(allProcessedData.items()),dtype=object)[:,1]
allProcessedLabel = np.asarray(list(allProcessedLabel.items()),dtype=object)[:,1]
deviceIndex =  np.asarray(list(deviceIndex.items()),dtype=object)[:,1]


# In[ ]:


deleteIndex = []
for index, i in enumerate(allProcessedLabel):
    if(len(np.unique(i)) < len(classCounts)):
        print("Removing client " + str(index))
        print(np.unique(i))
        deleteIndex.append(index)
        for key, value in dict(deviceIndexes).items():
            if(value.count(index)):
                value.remove(index)
allProcessedLabel = np.delete(allProcessedLabel, deleteIndex)
allProcessedData = np.delete(allProcessedData, deleteIndex)
deviceIndex = np.delete(deviceIndex, deleteIndex)


# In[ ]:


clientRange = [len(arrayLength) for arrayLength in allProcessedLabel]


# In[ ]:


deviceSize = []
for key, value in dict(deviceIndexes).items():
    deviceSize.append(len(value))


# In[ ]:


normalizedData = []


# In[ ]:


endIndex = 0 
for i in deviceSize:
    startIndex = endIndex
    endIndex += i
    deviceData = np.vstack(allProcessedData[startIndex:endIndex])
    deviceDataAcc = deviceData[:,:,:3].astype(np.float32)
    deviceDataGyro = deviceData[:,:,3:].astype(np.float32)
    accMean =  np.mean(deviceDataAcc)
    accStd =  np.std(deviceDataAcc)
    gyroMean =  np.mean(deviceDataGyro)
    gyroStd =  np.std(deviceDataGyro)
    deviceDataAcc = (deviceDataAcc - accMean)/accStd
    deviceDataGyro = (deviceDataGyro - gyroMean)/gyroStd
    deviceData = np.dstack((deviceDataAcc,deviceDataGyro))
    normalizedData.append(deviceData)


# In[ ]:


normalizedData = np.vstack(normalizedData)


# In[ ]:


startIndex = 0
endIndex = 0 
dataName = 'HHAR'
os.makedirs('datasetStandardized/'+dataName, exist_ok=True)
# clientRange
for i, dataRange in enumerate(clientRange):
    startIndex = endIndex 
    endIndex = startIndex + dataRange
    hkl.dump(normalizedData[startIndex:endIndex],'datasetStandardized/'+dataName+'/UserData'+str(i)+'.hkl' )
    hkl.dump(allProcessedLabel[i],'datasetStandardized/'+dataName+'/UserLabel'+str(i)+'.hkl' )
hkl.dump(deviceIndex,'datasetStandardized/'+dataName+'/deviceIndex.hkl' )


# In[ ]:


print("data processing finished")


# In[ ]:




