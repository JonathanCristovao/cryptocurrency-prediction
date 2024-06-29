

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, trainX, trainY, testX, testY):
        self.model.compile(optimizer='adam', loss='mse')
        history = self.model.fit(trainX, trainY, epochs=30, validation_data=(testX, testY), batch_size=64)
        return history


