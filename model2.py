from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.optimizers import Adam, SGD
from keras.metrics import top_k_categorical_accuracy
from sklearn import preprocessing


def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def compile_model():
    scaler = None
    model = Sequential()
    model.add(Dense(2048,
                    input_dim=2048,
                    activation='linear'
                    ))
    model.add(PReLU())
    model.add(Dense(8192, activation='linear'))
    model.add(PReLU())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_2])
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return model, scaler


if __name__ == '__main__':
    compile_model()
