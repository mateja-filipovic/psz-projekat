import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from assignments.utils.load_dataframe import load_dataframe_from_mysql_database

with open('utils/location_heuristics.json', 'r', encoding='utf-8') as file:
    heuristics = json.load(file)

location_filter = [
    'opština zvezdara',
    'opština vračar',
    'opština novi beograd',
    'opština savski venac',
    'opština stari grad',
]


def calculate_distance_from_city_centre(location):
    location_lower = location.lower()
    return heuristics[location_lower] if location_lower in heuristics else 3


def convert_floor_to_numeric(floor):
    if floor is None:
        return 0

    if isinstance(floor, (int, float)):
        return floor

    if floor.isdigit():
        return int(floor)
    if floor.lower() == 'vpr':
        return 0.5
    if floor.lower() == 'pr':
        return 0
    if floor.lower() == 'sut':
        return -1
    if floor.lower() == 'psut':
        return -0.5

    return 0


def convert_boolean_to_numeric(value):
    return 1 if value else 0


def convert_real_estate_type_to_numeric(real_estate_type):
    if real_estate_type is None:
        return 0
    return 1 if real_estate_type.lower() == 'novogradnja' else 0


def remove_outliers_percentage(df):
    cutoff_threshold_low = np.percentile(df['price'], 3)
    cutoff_threshold_high = np.percentile(df['price'], 97)

    return df[(df['price'] >= cutoff_threshold_low) & (df['price'] <= cutoff_threshold_high)]


def preprocess_dataframe(sell_only_data, user_prediction=False):
    result = pd.DataFrame()

    result['floor'] = sell_only_data['floor'].apply(convert_floor_to_numeric)
    result['has_lift'] = sell_only_data['has_lift'].apply(convert_boolean_to_numeric)
    result['has_parking'] = sell_only_data['has_parking'].apply(convert_boolean_to_numeric)
    result['is_registered'] = sell_only_data['is_registered'].apply(convert_boolean_to_numeric)
    result['real_estate_type'] = sell_only_data['real_estate_type'].apply(convert_real_estate_type_to_numeric)
    result['distance_to_center'] = sell_only_data['microlocation'].apply(calculate_distance_from_city_centre)
    result['real_estate_surface_area'] = sell_only_data['real_estate_surface_area']

    if not user_prediction:
        result['price'] = sell_only_data['price']
    if not user_prediction:
        result = remove_outliers_percentage(result)

    return result


class LinearRegression:
    def __init__(self, learning_rate, number_of_epochs):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.loss_history = None

    def step(self, errors):
        self.weights -= self.learning_rate * np.dot(self.X_train.T, errors)

    def loss(self, errors):
        return np.mean(errors ** 2)

    def _initialize_model(self, X_train, y_train):
        # add a column of ones, for w0
        X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))

        self.X_train = X_train
        self.y_train = y_train

        self.number_of_samples, self.number_of_features = X_train.shape
        self.number_of_features = self.number_of_features - 1

        # w0 and number of features more weights
        self.weights = np.random.rand(self.number_of_features + 1, 1)

        self.loss_history = []

    def fit(self, X_train, y_train):
        self._initialize_model(X_train, y_train)

        for i in range(self.number_of_epochs):
            predictions = np.dot(self.X_train, self.weights)
            errors = predictions - self.y_train
            self.step(errors)
            mse_loss = self.loss(errors)
            self.loss_history.append(mse_loss)
            print(f"Epoch: {i}/{self.number_of_epochs}, train loss = {mse_loss}")

    def predict(self, X_test):
        X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
        return np.dot(X_test, self.weights)

    def plot_loss_history(self):
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.show()


    def accuracy_score(self, y_test, y_predictions):
        return np.mean((y_test - y_predictions) ** 2)


data = load_dataframe_from_mysql_database()
pd.set_option('max_colwidth', None)

central_municipalities_only = data[data['location'].isin(location_filter)].reset_index()
apartments_only = central_municipalities_only[central_municipalities_only['real_estate_type'] == 'stan']
sell_only = apartments_only[apartments_only['transaction_type'] == 'prodaja']
preprocessed_data = preprocess_dataframe(sell_only)
preprocessed_data.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_data.drop('price', axis=1),
                                                    preprocessed_data['price'],
                                                    test_size=0.2,
                                                    random_state=42)


# feature scaling
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)
X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# label scaling
label_scaler = StandardScaler()
label_scaler.fit(y_train.values.reshape(-1, 1))
y_train_scaled = label_scaler.transform(y_train.values.reshape(-1, 1))
y_test_scaled = label_scaler.transform(y_test.values.reshape(-1, 1))


# training
linear_regression = LinearRegression(0.00001, 1000)
linear_regression.fit(X_train_scaled, y_train_scaled)

sklearn_model = LinReg()
sklearn_model.fit(X_train_scaled, y_train_scaled)

# testing
predictions = linear_regression.predict(X_test_scaled)
predictions_sklearn = sklearn_model.predict(X_test_scaled)

predictions_unscaled = label_scaler.inverse_transform(predictions).reshape(-1)
predictions_unscaled_sklearn = label_scaler.inverse_transform(predictions_sklearn).reshape(-1, 1)

for i in range(len(predictions_unscaled)):
    print("Predicted value:", predictions_unscaled[i])
    print("Sklearn predict:", predictions_unscaled_sklearn[i])
    print("Actual value:", y_test.values[i])

print("Accuracy score:", linear_regression.accuracy_score(y_test_scaled, predictions))

linear_regression.plot_loss_history()


# predictions for user input
def predict_for_user_input():
    floor = int(input('Input real estate floor: '))
    has_lift = input('Input lift status (False - no lift, True - has lift): ').lower() == 'true'
    has_parking = input('Input parking status (False - no parking, True - has parking): ').lower() == 'true'
    is_registered = input('Input registered status (False - not registered, True - is registered): ').lower() == 'true'
    real_estate_type = input('Input real estate type (novogradnja, stara gradnja): ')
    microlocation = input('Enter location within the city: ')
    real_estate_surface_area = float(input('Enter surface area: '))

    # convert to dataframe
    data = {
        'floor': [floor],
        'has_lift': [has_lift],
        'has_parking': [has_parking],
        'is_registered': [is_registered],
        'real_estate_type': [real_estate_type],
        'microlocation': [microlocation],
        'real_estate_surface_area': [real_estate_surface_area]
    }

    data_frame = pd.DataFrame(data)
    data_frame = preprocess_dataframe(data_frame, user_prediction=True)
    data_frame_scaled = feature_scaler.transform(data_frame)

    prediction = linear_regression.predict(data_frame_scaled)
    prediction = label_scaler.inverse_transform(prediction)
    print(f'Predicted value: {prediction}')


while True:
    option = input('Unesite 1 za predikciju; bilo sta drugo za izlaz')

    if option != '1':
        break

    predict_for_user_input()


