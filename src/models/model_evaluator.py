from numpy import sqrt
from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def evaluate(self, testX, testY):
        predicted = self.model.predict(testX)
        predicted_inverse = self.scaler.inverse_transform(predicted)
        testY_2D = testY.reshape(-1, 1)
        testY_inverse = self.scaler.inverse_transform(testY_2D)
        rmse = sqrt(mean_squared_error(testY_inverse, predicted_inverse))
        print('Test RMSE:', rmse)
        return predicted_inverse, testY_inverse
    
    def predict_next_day(self, last_sequence):
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        next_day_prediction = self.model.predict(last_sequence)
        next_day_prediction_inverse = self.scaler.inverse_transform(next_day_prediction)
        return next_day_prediction, next_day_prediction_inverse


