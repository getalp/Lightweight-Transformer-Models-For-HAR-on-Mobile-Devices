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
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# In[ ]:


fileName = [
        "SHLDataset_preview_v1_part1.zip",
        "SHLDataset_preview_v1_part2.zip",
        "SHLDataset_preview_v1_part3.zip"
           ]
links = [
    "http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part1.zip",
        "http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part2.zip",
        "http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part3.zip"]


# In[ ]:


# # download and unzipping dataset/download
# os.makedirs('dataset/download',exist_ok=True)
# os.makedirs('dataset/extracted',exist_ok=True)

# for i in range(len(fileName)):
#     data_directory = os.path.abspath("dataset/download/"+str(fileName[i]))
#     if not os.path.exists(data_directory):
#         print("downloading "+str(fileName[i]))            
#         download_url(links[i],data_directory)
#         print("download done")
#         data_directory2 =  os.path.abspath("dataset/extracted/"+str(fileName[i]))
#         print("extracting data...")
#         with zipfile.ZipFile(data_directory, 'r') as zip_ref:
#             zip_ref.extractall(os.path.abspath("dataset/extracted/"))
#         print("data extracted")
#     else:
#         print(str(fileName[i]) + " already downloaded")


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


bodyLocations = ["Bag","Hand","Hips","Torso"] 
rootDirectory = 'dataset/extracted/SHLDataset_preview_v1'
dirs = [d for d in os.listdir(rootDirectory) if os.path.isdir(os.path.join(rootDirectory, d))]
dirs.remove("scripts")


# In[ ]:


def processLabel(labels):
    uniqueCount = np.unique(labels,return_counts=True)
    if(len(uniqueCount[0]) > 1):
        return uniqueCount[0][np.argmax(uniqueCount[1])]
    else:
        return uniqueCount[0][0]


# In[ ]:


def segmentData(accData,time_step,step):
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(accData[i:i+time_step,:])


    return np.asarray(segmentAccData)
def segmentLabel(accData,time_step,step):
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(processLabel(accData[i:i+time_step]))
        
    return np.asarray(segmentAccData)


# In[ ]:


def downSampleLowPass(motionData):
    accX = signal.decimate(motionData[:,:,0],2)
    accY = signal.decimate(motionData[:,:,1],2)
    accZ = signal.decimate(motionData[:,:,2],2)
    gyroX = signal.decimate(motionData[:,:,3],2)
    gyroY = signal.decimate(motionData[:,:,4],2)
    gyroZ = signal.decimate(motionData[:,:,5],2)
    return np.dstack((accX,accY,accZ,gyroX,gyroY,gyroZ))


# In[ ]:


print("processing data...")
userData = []
userLabel = []
for userSubFolder in dirs:
    print("procesing "+userSubFolder)
    subDir =  rootDirectory + "/"+userSubFolder
    timeSubFolder = [d for d in os.listdir(subDir) if os.path.isdir(os.path.join(subDir, d))]

    start = True
    for timeDir in timeSubFolder:
        dataPosition = []
        dataDir = subDir + "/" + timeDir
        labelPosition = load_file(dataDir+"/Label.txt")[:,1]
        nanSortedLocation = []
        for locations in bodyLocations:
            dataPosition.append(load_file(dataDir+"/"+locations +"_Motion.txt")[:,1:7])
            nanlocation = findRanges(np.unique(np.where(np.isnan(dataPosition[-1]))))
            for eachRange in nanlocation:
                nanSortedLocation.append(eachRange)
        deleteRange = unionRange(nanSortedLocation)
        for i in reversed(range(len(deleteRange))):
            labelPosition = np.delete(labelPosition,np.s_[deleteRange[i][0]:deleteRange[i][1]],axis=0)
            for bodyCount in range(len(bodyLocations)):
                dataPosition[bodyCount]  = np.delete(dataPosition[bodyCount],np.s_[deleteRange[i][0]:deleteRange[i][1]],axis=0)
                
#         segmenting data and removing null class frames       
        labelPosition = segmentLabel(labelPosition,256,128) 
        nullFrameToDelete = np.where(labelPosition == 0 )
        labelPosition = np.delete(labelPosition,nullFrameToDelete)
        labelPosition = labelPosition - 1
        labelPosition = np.swapaxes(np.repeat(labelPosition[:,  np.newaxis], len(bodyLocations), axis=1),0,1)
        for bodyCount in range(len(bodyLocations)):
            dataPosition[bodyCount] = segmentData(dataPosition[bodyCount],256,128)
            dataPosition[bodyCount] = np.delete(dataPosition[bodyCount],nullFrameToDelete,axis=0)
            dataPosition[bodyCount] = downSampleLowPass(dataPosition[bodyCount])
        userData.append(dataPosition)
        userLabel.append(labelPosition)


# In[ ]:


combinedUserData = np.hstack((userData))


# In[ ]:


combinedAccData = combinedUserData[:,:,:,:3]
combinedGyroData = combinedUserData[:,:,:,3:]


# In[ ]:


accMean =  np.mean(combinedAccData)
accStd =  np.std(combinedAccData)
                   
gyroMean =  np.mean(combinedGyroData)
gyroStd =  np.std(combinedGyroData)


# In[ ]:


userData = np.asarray(userData, dtype=objec)


# In[ ]:


labels = []
for clientIndex in range(userData.shape[0]):
    for bodyIndex in range(userData.shape[1]):
        userData[clientIndex][bodyIndex][:,:,:3] = (userData[clientIndex][bodyIndex][:,:,:3] - accMean)/accStd
        userData[clientIndex][bodyIndex][:,:,3:] = (userData[clientIndex][bodyIndex][:,:,3:] - gyroMean)/gyroStd
        labels.append(userLabel[clientIndex][bodyIndex])
labels = np.asarray(labels, dtype=objec)
userData = np.hstack((userData))


# In[ ]:


startIndex = 0
endIndex = 0 
dataName = 'SHL'
os.makedirs('datasetStandardized/'+dataName, exist_ok=True)
hkl.dump(userData,'datasetStandardized/'+dataName+'/clientsData.hkl' )
hkl.dump(labels,'datasetStandardized/'+dataName+'/clientsLabel.hkl' )


# In[ ]:


print("data processing finished")

