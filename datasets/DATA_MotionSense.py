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
import tensorflow as tf


# In[ ]:


# functions for loading and downloading the dataset

# load a single file as a numpy array
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, sep = ',')
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


def matchLabel(activityFileName):
    if "dws" in activityFileName:
        return 0
    elif "ups" in activityFileName:
        return 1
    elif "sit" in activityFileName:
        return 2
    elif "std" in activityFileName:
        return 3
    elif "wlk" in activityFileName:
        return 4
    elif "jog" in activityFileName:
        return 5
    else:
        print("Not found!")
        return None


# In[ ]:


def segmentData(accData,time_step,step):
#     print(accData.shape)
    step = int(step)
    segmentAccData = []
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(accData[i:i+time_step,:])
    return np.asarray(segmentAccData)


# In[ ]:


fileName = ["A_DeviceMotion_data"]
links = ["https://github.com/mmalekzadeh/motion-sense/blob/master/data/A_DeviceMotion_data.zip?raw=true"]


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
        data_directory2 =  os.path.abspath("dataset/extracted/"+str(fileName[i])+".zip")
        print("extracting data...")
        with zipfile.ZipFile(data_directory, 'r') as zip_ref:
            zip_ref.extractall(os.path.abspath("dataset/extracted/"))
        print("data extracted")
    else:
        print(str(fileName[i]) + " already downloaded")


# In[ ]:


extractedDirectory = os.path.abspath("dataset/extracted/A_DeviceMotion_data")
dirs = np.sort([d for d in os.listdir(extractedDirectory) if os.path.isdir(os.path.join(extractedDirectory, d))])


# In[ ]:


clientData = {iniArray: [] for iniArray in range(24)}
clientLabel = {iniArray: [] for iniArray in range(24)}


# In[ ]:


for activityIndex, activityFileName in enumerate(dirs):
    subjectFileNames = sorted(os.listdir(extractedDirectory + "/"+activityFileName))
    for clientIndex, subFileName in enumerate(subjectFileNames):
        loadedData = load_file(extractedDirectory+"/"+activityFileName+"/"+subFileName)
        processedData = segmentData(np.hstack((loadedData[:,10:13],loadedData[:,7:10]))[1:,:],128,64)
        clientData[clientIndex].append(processedData.astype(np.float32))
        clientLabel[clientIndex].append(np.full(processedData.shape[0], matchLabel(activityFileName), dtype=int))


# In[ ]:


processedData = []
processedLabel = []
clientSize = []
for clientIndex in range(24):
    processedData.append(np.vstack((clientData[clientIndex])))
    processedLabel.append(np.hstack((clientLabel[clientIndex])))


# In[ ]:


combinedUserData = np.vstack((processedData))


# In[ ]:


combinedAccData = combinedUserData[:,:,:3]
combinedGyroData = combinedUserData[:,:,3:]


# In[ ]:


accMean =  np.mean(combinedAccData)
accStd =  np.std(combinedAccData)
                   
gyroMean =  np.mean(combinedGyroData)
gyroStd =  np.std(combinedGyroData)


# In[ ]:


combinedAccData = (combinedAccData - accMean)/accStd
combinedGyroData = (combinedGyroData - gyroMean)/gyroStd


# In[ ]:


combinedUserData = np.dstack((combinedAccData,combinedGyroData))


# In[ ]:


startIndex = 0
endIndex = 0 
dataName = 'MotionSense'
os.makedirs('datasetStandardized/'+dataName, exist_ok=True)
for i in range(len(processedData)):
    startIndex = endIndex 
    endIndex = startIndex +  processedData[i].shape[0]
    hkl.dump(combinedUserData[startIndex:endIndex],'datasetStandardized/'+dataName+'/UserData'+str(i)+'.hkl' )
    hkl.dump(processedLabel[i],'datasetStandardized/'+dataName+'/UserLabel'+str(i)+'.hkl' )


# In[ ]:


print("data processing finished")


# In[ ]:




