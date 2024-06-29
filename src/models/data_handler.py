import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataHandler:
    def __init__(self, ticker='BTC-USD', start_date='2010-01-01', end_date='2022-01-01'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    
    def load_and_preprocess_data(self):
        self.load_data()
        return self.preprocess_data()

    
    def load_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data.dropna(inplace=True)
        return self.data

    def preprocess_data(self):
        self.data['Weighted_Price'] = (self.data['Open'] + self.data['High'] + self.data['Low'] + self.data['Close']) / 4
        values = self.data['Weighted_Price'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled = scaler.fit_transform(values)
        return self.scaled, scaler

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        dataX = np.array(dataX)
        dataX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
        return dataX, np.array(dataY)