from dataset.ml_dataset import *
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Vanilla_MF:
    def __init__(self):
        self.user_track_matrix = userTrackMatrix()
        self.user_track_matrix.load()
        self.MF_users = self.user_track_matrix.MF_users
        self.MF_tracks = self.user_track_matrix.MF_tracks

    def predict(self, user):
        scores = np.dot(self.MF_users[user, :], self.MF_tracks)
        positions = np.argsort(scores)[::-1]
        return positions

