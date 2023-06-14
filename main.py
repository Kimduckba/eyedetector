'''
-> 이 코드 실행해야. from import 사용 가능.
from google.colab import drive
drive.mount('/content/drive')

'''

#1
# 68개 얼굴 랜드마크 데이터 가져오기
! wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 압축 풀기
! bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

import sys
sys.path.append('/content/drive/MyDrive/Eye/import')
from EYE_Model import EYE_Model
from EyeDataset import EyeDataset
from EyeDetector import EyeDetector
from Graph import Graph

import numpy as np
import os, cv2, dlib
from imutils import face_utils
import matplotlib.pyplot as plt
from dataclasses import dataclass
from keras.models import load_model
from keras.utils import to_categorical
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split

img_row = 26
img_col = 34
ch = 1

data_path = '/content'

eye_dataset = EyeDataset(data_path=data_path, img_row=img_row, img_col=img_col, ch=ch)
Train_Imgs, Train_Labels, Val_Imgs, Val_Labels, Test_Imgs, Test_Labels = eye_dataset.load_data()

epochs = 3000

# 모델을 생성하고 훈련합니다.
eye_model = EYE_Model(
    img_row=img_row,
    img_col=img_col,
    ch=ch,
    epochs= epochs,
    Train_Imgs=Train_Imgs,
    Train_Labels=Train_Labels,
    Val_Imgs=Val_Imgs,
    Val_Labels=Val_Labels,
    Test_Imgs=Test_Imgs,
    Test_Labels=Test_Labels
)

history = eye_model.train()

# 예측을 수행합니다.
y_pred = eye_model.predict(Test_Imgs)

# 이전 단계에서 얻은 history 변수를 전달합니다.
graph = Graph(history)

# 손실과 정확도 그래프를 그립니다.
graph.plot_loss_and_accuracy()
