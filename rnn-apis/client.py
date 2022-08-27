import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import librosa as lib
from sklearn import preprocessing
import os
import requests
import shutil

MODEL_PATH = r"C:\Users\sriram\Desktop\rnn-apis\gru-bimodel_V.1.h5"
model = load_model(MODEL_PATH)

dir = r"C:\Users\sriram\Desktop\rnn-apis\sample_files"
new_dir = r"C:\Users\sriram\Desktop\rnn-apis\moved_files"

def compute_mfcc(audio, rate):
    mfcc_feature = lib.feature.mfcc(y= audio, sr= rate, n_mfcc = 20 ,hop_length = 256 ,n_fft = 1024)
    #n_mfcc gives no. of mfcc coefficients required,hop_length gives no. of samples between successive frames,n_fft gives length of the fft operation
    mfcc_feature = preprocessing.scale(mfcc_feature , axis = 1)
    mfcc_feature = mfcc_feature.T
#     f = plt.figure(1, figsize = (25,10))
#     f.set_figwidth(8)
#     f.set_figheight(4)
#     librosa.display.specshow(mfcc_feature , x_axis = 'time' , sr = (0.025 * rate))
#     plt.colorbar(format='%+2f')
#     plt.title("MFCC")
#     plt.show()
#     plt.pause(1)
#     plt.close()
    #print(mfcc_feature.shape)
    return mfcc_feature


activities = ["Movement" , "No_Movement"]
for root, _, files in os.walk(dir):
    for file in files:
        file_path = dir + '\\' + file
        print(file_path)
        data = pd.read_csv(file_path, skiprows=22, header = None).iloc[: , 2]
        mfcc = compute_mfcc(data.to_numpy() , 8000)
        mfcc = mfcc.reshape((1,32,20))
        predict = model.predict(mfcc)
        result = activities[np.argmax(predict[0])]
        if result == "No_Movement":
            print("No_Movement")
            shutil.move(file_path , new_dir+'\\'+file)
        else:
            print("Movement")
            r = requests.post("http://127.0.0.1:5000/get_file", files = {'file' : open(file_path , 'rb')})
            print(r.status_code)

            shutil.move(file_path , new_dir+'\\'+file)



