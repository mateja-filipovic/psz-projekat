import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def convert_price_to_price_category(price):
    if price < 50000:
        return 'expensive'
    elif price < 100000:
        return '50000<100000'
    elif price < 150000:
        return '100000<150000'
    elif price < 200000:
        return '150000<200000'
    elif price < 500000:
        return '200000<500000'
    else:
        return '50000+'


def preprocess_dataframe(sell_only_data, user_prediction=False):
    result = pd.DataFrame()

    result['floor'] = sell_only_data['floor'].apply(convert_floor_to_numeric)
    result['real_estate_type'] = sell_only_data['real_estate_type'].apply(convert_real_estate_type_to_numeric)
    result['distance_to_center'] = sell_only_data['microlocation'].apply(calculate_distance_from_city_centre)
    result['real_estate_surface_area'] = sell_only_data['real_estate_surface_area']

    if not user_prediction:
        result['price_category'] = sell_only_data['price'].apply(convert_price_to_price_category)

    return result


class KNNClassifier:
    def __init__(self, number_of_neighbors):
        self.X_train = None
        self.y_train = None
        self.number_of_neighbors = number_of_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.values

    def distance_euclidian(self, x1, x2):
        result = (x1 - x2) ** 2
        result = np.sum(result)
        result = math.sqrt(result)

        return result

    def distance_manhattan(self, x1, x2):
        result = np.abs(x1 - x2)
        result = sum(result)
        return result

    def predict_for_single_datapoint(self, test_row):
        distances = np.empty((self.X_train.shape[0],))

        for train_index, train_row in enumerate(self.X_train):
            distance = self.distance_euclidian(test_row, train_row)
            distances[train_index] = distance

        closest_neighbors_indexes = np.argsort(distances)[:self.number_of_neighbors]
        closest_neighbors_classes = self.y_train[closest_neighbors_indexes]

        unique_classes, unique_classes_occurrences = np.unique(closest_neighbors_classes, return_counts=True)
        most_common_class = unique_classes[np.argmax(unique_classes_occurrences)]

        return most_common_class

    def predict(self, x_test):
        predictions = []
        for test_index, test_row in enumerate(x_test):
            prediction = self.predict_for_single_datapoint(test_row)
            predictions.append(prediction)
        return predictions

    def accuracy_score(self, y_test, y_predictions):
        number_of_correct_predictions = np.sum(y_predictions == y_test)
        number_of_samples = len(y_test)
        return number_of_correct_predictions / number_of_samples


data = load_dataframe_from_mysql_database()
pd.set_option('max_colwidth', None)

central_municipalities_only = data[data['location'].isin(location_filter)].reset_index()
apartments_only = central_municipalities_only[central_municipalities_only['real_estate_type'] == 'stan']
sell_only = apartments_only[apartments_only['transaction_type'] == 'prodaja']


preprocessed_data = preprocess_dataframe(sell_only)
preprocessed_data.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_data.drop('price_category', axis=1),
                                                    preprocessed_data['price_category'],
                                                    test_size=0.2,
                                                    random_state=42)


# hyper parameters
number_of_neighbors = 5

# feature scaling
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)
X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# training
classifier = KNNClassifier(number_of_neighbors)
classifier.fit(X_train_scaled, y_train)

# predictions
predictions = classifier.predict(X_test_scaled)
model_accuracy = classifier.accuracy_score(y_test.values, predictions)
print(model_accuracy)


# predictions for user input
def predict_for_user_input():
    floor = int(input('Input real estate floor: '))
    real_estate_type = input('Input real estate type (novogradnja, stara gradnja): ')
    microlocation = input('Enter location within the city: ')
    real_estate_surface_area = float(input('Enter surface area: '))

    # convert to dataframe
    data = {
        'floor': [floor],
        'real_estate_type': [real_estate_type],
        'microlocation': [microlocation],
        'real_estate_surface_area': [real_estate_surface_area]
    }

    data_frame = pd.DataFrame(data)
    data_frame = preprocess_dataframe(data_frame, user_prediction=True)
    data_frame_scaled = feature_scaler.transform(data_frame)

    prediction = classifier.predict(data_frame_scaled)
    print(f'Predicted value: {prediction}')


while True:
    option = input('Unesite 1 za predikciju; bilo sta drugo za izlaz')

    if option != '1':
        break

    predict_for_user_input()
