from dataset.ml_dataset import *
from sklearn.neighbors import NearestNeighbors
import numpy as np

class BaseVanilla:
    def __init__(self):
        self.user_track_matrix = userTrackMatrix()
        self.user_track_matrix.load()
        self.MF_users = self.user_track_matrix.MF_users
        self.MF_tracks = self.user_track_matrix.MF_tracks


class Vanilla_MF(BaseVanilla):
    def __init__(self):
        super().__init__()

    def predict(self, user):
        scores = np.dot(self.MF_users[user, :], self.MF_tracks)
        positions = np.argsort(scores)[::-1]
        return positions

class Vanilla_NN(BaseVanilla):
    def __init__(self, K = 1):
        super().__init__()
        self.K = K
        self.NN = NearestNeighbors(n_neighbors=self.K + 1, algorithm='ball_tree')
        self.NN.fit(self.MF_users)

    def get_nearest_users(self, user_id, k_nearest):
        """ finds for a given user the most similar user, and returns its songs as predictions """
        X = self.MF_users



