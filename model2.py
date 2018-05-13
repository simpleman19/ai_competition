from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.optimizers import Adam, SGD
from sklearn import preprocessing


def compile_model():
    scaler = None
    model = Sequential()
    model.add(Dense(2048,
                    input_dim=2048,
                    activation='tanh'
                    ))
    model.add(PReLU())
    model.add(Dense(4096, activation='linear'))
    model.add(PReLU())
    model.add(Dense(2048, activation='linear'))
    model.add(PReLU())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return model, scaler


if __name__ == '__main__':
    compile_model()
