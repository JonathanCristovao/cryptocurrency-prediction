import datetime
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

class Visualization:
    @staticmethod
    def plot_results(dates, predicted_data, true_data, next_day_prediction=None):
        plt.figure(figsize=(14, 7))
        
        # Certifique-se de que as datas e os dados correspondem em tamanho
        if len(dates) != len(true_data) or len(dates) != len(predicted_data):
            raise ValueError("As listas de datas, dados verdadeiros e dados preditos devem ter o mesmo tamanho.")
        
        plt.plot(dates, true_data, label='True Data', color='blue', marker='o', linestyle='-')
        plt.plot(dates, predicted_data, label='Model Prediction', color='red', marker='', linestyle='--')

        if next_day_prediction is not None:
            # Calcula a data do próximo dia
            next_day_date = dates[-1] + datetime.timedelta(days=1)
            plt.scatter([next_day_date], [next_day_prediction], color='green', marker='x', s=100, label='Next Day Prediction')
            plt.axvline(x=next_day_date, color='green', linestyle='--', label='Prediction Date')

        plt.title('Prediction vs True Data')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))  # Ajuste conforme necessário
        plt.gcf().autofmt_xdate()  # Rotação automática das datas
        plt.show()
