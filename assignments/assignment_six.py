import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from assignments.utils.load_dataframe import load_dataframe_from_mysql_database

location_filter = [
    'opština zvezdara',
    'opština vračar',
    'opština novi beograd',
    'opština savski venac',
    'opština stari grad',
]


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


def preprocess_dataframe(sell_only_data):
    result = pd.DataFrame()

    result['floor'] = sell_only_data['floor'].apply(convert_floor_to_numeric)
    result['has_lift'] = sell_only_data['has_lift'].apply(convert_boolean_to_numeric)
    result['has_parking'] = sell_only_data['has_parking'].apply(convert_boolean_to_numeric)
    result['is_registered'] = sell_only_data['is_registered'].apply(convert_boolean_to_numeric)
    result['real_estate_type'] = sell_only_data['real_estate_type'].apply(convert_real_estate_type_to_numeric)
    result['real_estate_surface_area'] = sell_only_data['real_estate_surface_area']
    result['price'] = sell_only_data['price']

    return result


class KMeansClustering:

    def __init__(self, number_of_clusters, max_iterations):
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations

        self.cluster_centers = None
        self.labels = None

    def distance_euclidean(self, x1, x2):
        result = (x1 - x2) ** 2
        result = np.sum(result)
        result = math.sqrt(result)

        return result

    def find_closest_cluster_center(self, X_row):
        distances = np.empty((len(self.cluster_centers),))

        for i in range(len(self.cluster_centers)):
            distances[i] = self.distance_euclidean(X_row, self.cluster_centers[i])

        closest_centroid_index = np.argmin(distances)
        return closest_centroid_index

    def update_cluster_centers(self, X):
        new_cluster_centers = np.empty((self.number_of_clusters, X.shape[1]))

        for i in range(self.number_of_clusters):
            cluster_points = X[self.labels == i]
            new_cluster_center = np.mean(cluster_points, axis=0)
            new_cluster_centers[i] = new_cluster_center

        self.cluster_centers = new_cluster_centers

    def fit(self, X):
        random_indices = np.random.choice(len(X), self.number_of_clusters)

        self.cluster_centers = X[random_indices]
        self.labels = np.empty((len(X),))

        for i in range(self.max_iterations):
            new_labels = np.empty((len(X),))

            # assign labels based on closest cluster center
            for index, row in enumerate(X):
                data_point_label = self.find_closest_cluster_center(row)
                new_labels[index] = data_point_label

            # break if no changes to labels
            if np.all(self.labels == new_labels):
                break

            self.labels = new_labels
            self.update_cluster_centers(X)


data = load_dataframe_from_mysql_database()
pd.set_option('max_colwidth', None)

central_municipalities_only = data[data['location'].isin(location_filter)].reset_index()
apartments_only = central_municipalities_only[central_municipalities_only['real_estate_type'] == 'stan']
sell_only = apartments_only[apartments_only['transaction_type'] == 'prodaja']


preprocessed_data = preprocess_dataframe(sell_only)
preprocessed_data.dropna(inplace=True)

data = preprocessed_data

# hyper parameters
number_of_clusters = 5

# feature scaling
feature_scaler = StandardScaler()
feature_scaler.fit(data)
X_train_scaled = feature_scaler.transform(data)

# training
k_means_clustering = KMeansClustering(number_of_clusters, 100000)
k_means_clustering.fit(X_train_scaled)

labels = k_means_clustering.labels
cluster_centers = k_means_clustering.cluster_centers

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['real_estate_surface_area'].values, data['has_lift'].values, data['floor'].values, c=labels, cmap='viridis')
ax.scatter(cluster_centers[:, 5], cluster_centers[:, 1], cluster_centers[:, 0], marker='x', color='red')
ax.set_xlabel('real_estate_surface_area')
ax.set_ylabel('has_lift')
ax.set_zlabel('floor')
plt.title('clusters')

plt.show()

