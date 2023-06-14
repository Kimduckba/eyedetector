import os
import numpy as np
from dataclasses import dataclass
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

@dataclass
class EyeDataset:
    data_path: str
    img_row: int
    img_col: int
    ch: int

    def load_data(self):
        x_train = np.load(os.path.join(self.data_path, 'Train_imgs.npy'))
        y_train = np.load(os.path.join(self.data_path, 'Train_labels.npy'))
        x_val = np.load(os.path.join(self.data_path, 'Test_imgs.npy'))
        y_val = np.load(os.path.join(self.data_path, 'Test_labels.npy'))

        #print(x_train.shape, y_train.shape) # (217, 26, 34) (217,)
        #print(x_val.shape, y_val.shape)     # (55, 26, 34) (55,)

        x_train = x_train.reshape(x_train.shape[0], self.img_row, self.img_col, self.ch)
        x_val = x_val.reshape(x_val.shape[0], self.img_row, self.img_col, self.ch)
        #print(x_train.shape) #  (217, 26, 34, 1)
        #print(y_train.shape) #  (217,)
        #print(x_val.shape)   #  (55, 26, 34, 1)
        #print(y_val.shape)   #  (55,)

        y_train = to_categorical(y_train, 2)
        y_val = to_categorical(y_val, 2)

        #print(y_train.shape) #  (217, 2)
        #print(y_val.shape)   #  (55, 2)

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        #print(x_train.shape)  #  (173, 26, 34, 1)
        #print(y_train.shape)  #  (173, 2)
        #print(x_test.shape)   #  (44, 26, 34, 1)
        #print(y_test.shape)   #  (44, 2)

        return x_train, y_train, x_val, y_val, x_test, y_test
