class predictor():

    def __init__(self, scaler, regressor):
        self.scaler = scaler
        self.regressor = regressor
        self.data = None
        self.min_date = None
        self.max_date = None
        self.mape_train = None
        self.mape_test = None
        self.feature_importance = None
        self.max_train_size = None
        self.text_size = None
        self.train_date = None

    def predict(self):
        scaled_data = self.scaler.transform(self.data)
        response = self.regressor.predict(scaled_data)
        return response
