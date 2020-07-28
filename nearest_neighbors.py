### Code here ###

import numpy as np
# from pybaseball import statcast, batting_stats

# Class
# describe code

class nearest_neighbors:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, a, b):
        eucl_dist = 0.0
        for index in range(len(a)):
            eucl_dist += (a[index] - b[index]) ** 2
            euclidean_distance = np.sqrt(eucl_dist)
        return euclidean_distance

    def fit_nn(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_nn(self, X):
        predict_nn = []
        for index in range(len(X)):
            euclidean_distances = []
            for row in self.X_train:
                eucl_dist = self.euclidean_distance(row, X[index])
                euclidean_distances.append(eucl_dist)
            
            neighbors = np.array(euclidean_distances).argsort()[: self.n_neighbors]
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_train[val]] = 1

                predict_nn.append(max(count_neighbors, key=count_neighbors.get))

            return predict_nn

    def display_nn(self, x):
        euclidean_distances = []

        for row in self.X_train:
            eucl_dist = self.euclidean_distance(row, x)
            euclidean_distances.append(eucl_dist)

        neighbors = np.array(euclidean_distances).argsort()[: self.n_neighbors]

        display_nn_values = []

        for index in range(len(neighbors)):
            neighbor_index = neighbors[index]
            e_dist = euclidean_distances[index]
            display_nn_values.append((neighbor_index, e_dist))

        return display_nn_values

# df = batting_stats(2019, league='all')
# df.head()

print('Running smooth!')