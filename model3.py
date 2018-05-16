from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam, SGD
from keras.metrics import top_k_categorical_accuracy
from keras import regularizers


def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def compile_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(1024, 2)))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_2])
    return model


if __name__ == '__main__':
    compile_model()
