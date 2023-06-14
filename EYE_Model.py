from dataclasses import dataclass
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

@dataclass
class EYE_Model:
    img_row: int
    img_col: int
    ch: int
    epochs: int
    Train_Imgs: int
    Train_Labels: int
    Val_Imgs: int
    Val_Labels: int
    Test_Imgs: int
    Test_Labels: int
    epochs: int

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (3,3), activation='relu', input_shape=(self.img_row, self.img_col, self.ch)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(16, (3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        model = self.build_model()
        print(model.summary())
        print('model fit...')
        history = model.fit(self.Train_Imgs, self.Train_Labels, batch_size=32, validation_data=(self.Val_Imgs, self.Val_Labels),epochs=self.epochs)
        score = model.evaluate(self.Test_Imgs, self.Test_Labels, self.epochs, verbose=0)
        print("%.2f%%" % (score[1] * 100))

        # model save
        # model.save('/content/drive/MyDrive/EYE/eye_model.h5')

        return history

    def predict(self, Test_Imgs):
        model = self.build_model()
        
        # 이미지 처리 및 예측 로직 추가
        
        y_pred = model.predict(Test_Imgs)

        return y_pred
