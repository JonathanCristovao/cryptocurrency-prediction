import datetime
from src.models.data_handler import DataHandler
from src.models.model_builder import ModelBuilder
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.views.visualization import Visualization

class PredictionController:
    def __init__(self):
        self.acoes_dict = {
            1: "BTC-USD",
            2: "ETH-USD",
            3: "BNB-USD",
            4: "ADA-USD",
            5: "SOL-USD",
            6: "XRP-USD",
            7: "LTC-USD",
        }
        
        self.dataHandler = DataHandler()
        self.modelBuilder = ModelBuilder()
        self.modelEvaluator = None
        self.modelTrainer = None
        self.visualization = Visualization()
        self.ticker_escolhido = None

    def escolher_acao(self):
        while True:
            print("\nSelecione uma ação para treinar o modelo ou digite 'q' para sair:")
            for numero, ticker in self.acoes_dict.items():
                print(f"{numero}: {ticker}")
            escolha = input("Digite o número correspondente à ação ou 'q' para sair: ")
            
            if escolha.lower() == 'q':
                print("Saindo do programa...")
                return None
            try:
                escolha_numero = int(escolha)
                if escolha_numero in self.acoes_dict:
                    self.ticker_escolhido = self.acoes_dict[escolha_numero]
                    return self.ticker_escolhido
                else:
                    print("Número fora das opções disponíveis. Tente novamente.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número ou 'q' para sair.")

    def run(self):
        ticker_escolhido = self.escolher_acao()
        if ticker_escolhido:
            print(f"Você escolheu treinar o modelo para: {ticker_escolhido}")
            next_day_actual = datetime.date.today()
            first_date = next_day_actual - datetime.timedelta(days=1000)
            next_day_prediction_date = next_day_actual + datetime.timedelta(days=1)
            
            self.dataHandler = DataHandler(ticker_escolhido, first_date, next_day_actual)
            data = self.dataHandler.load_data()
            scaled_data, scaler = self.dataHandler.preprocess_data()

            train_size = int(len(scaled_data) * 0.7)
            train, test = scaled_data[:train_size], scaled_data[train_size:]
            trainX, trainY = self.dataHandler.create_dataset(train)
            testX, testY = self.dataHandler.create_dataset(test)

            self.model = self.modelBuilder.build_model((trainX.shape[1], trainX.shape[2]))
            self.modelTrainer = ModelTrainer(self.model)
            history = self.modelTrainer.train(trainX, trainY, testX, testY)

            self.modelEvaluator = ModelEvaluator(self.model, scaler)
            predicted, true = self.modelEvaluator.evaluate(testX, testY)
            last_sequence = testX[-1]
            next_day_prediction, next_day_prediction_inverse = self.modelEvaluator.predict_next_day(last_sequence)
            next_day_prediction_plot = next_day_prediction_inverse.flatten()[0]

            dates = [first_date + datetime.timedelta(days=x) for x in range((next_day_prediction_date - first_date).days + 1)]
            min_length = min(len(dates), len(predicted), len(true))
            dates = dates[:min_length]
            predicted = predicted[:min_length]
            true = true[:min_length]

            self.visualization.plot_results(dates, predicted.flatten(), true.flatten(), next_day_prediction_plot)
