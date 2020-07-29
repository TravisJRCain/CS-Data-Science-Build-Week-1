# Imports
import numpy as np


# Class
class k_nearest_neighbors:
 # Initialization
    def __init__(self, n_neighbors=5):
        """Init for algorithm"""
        self.n_neighbors = n_neighbors
        
# Euclidean distance
    def euclidean_distance(self, a, b):
        eucl_distance = 0.0  # initializing eucl_distance at 0

        for index in range(len(a)):
            eucl_distance += (a[index] - b[index]) ** 2

            euclidian_distance = np.sqrt(eucl_distance)

        return euclidian_distance

    # Fit k Nearest Neighbors
    def fit_knn(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Predict X for kNN
    def predict_knn(self, X):

        # initialize prediction_knn as empty list
        prediction_knn = []

        for index in range(len(X)):  # Main loop iterating through len(X)

            # initialize euclidean_distance as empty list
            euclidean_distance = []

            for row in self.X_train:
                # for every row in X_train, find eucl_distance to X using
                # euclidean_distance() and append to euclidean_distance list
                eucl_distance = self.euclidean_distance(row, X[index])
                euclidean_distance.append(eucl_distance)

            neighbors = np.array(euclidean_distance).argsort()[: self.n_neighbors]

            # initialize dict to count class occurrences in y_train
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_train[val]] = 1

            # max count labels to prediction_knn
            prediction_knn.append(max(count_neighbors, key=count_neighbors.get))

        return prediction_knn

    # Print/display list of nearest_neighbors + corresponding euclidian
    # distance
    def display_knn(self, x):

        # initialize euclidean_distance as empty list
        euclidean_distance = []

        for row in self.X_train:
            eucl_distance = self.euclidean_distance(row, x)
            euclidean_distance.append(eucl_distance)

        neighbors = np.array(euclidean_distance).argsort()[: self.n_neighbors]

        # initiate empty display_knn_values list
        display_knn_values = []

        for index in range(len(neighbors)):
            neighbor_index = neighbors[index]
            e_distances = euclidean_distance[index]
            display_knn_values.append(
                (neighbor_index, e_distances)
            )  # changed to list of tuples
        # print(display_knn_values)
        return display_knn_values

# df = batting_stats(2019, league='all')
# df.head()

# print('Running smooth!')