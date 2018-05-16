from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU, ELU, Conv2D
from keras.backend import relu
from keras.optimizers import Adam, SGD
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing


def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def compile_model():
    scaler = None
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(2, 20), strides=(0, 1), activation='relu', input_shape=(1024, 2)))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_2])
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return model, scaler


if __name__ == '__main__':
    compile_model()
