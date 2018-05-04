from keras.models import Sequential
from keras.layers import Dense


def compile_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
