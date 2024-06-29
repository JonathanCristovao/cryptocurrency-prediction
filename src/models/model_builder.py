from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional

class ModelBuilder:
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(16))
        model.add(Dense(1))
        return model


