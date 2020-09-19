from dataset.user_item_matrix import *
from sklearn.neighbors import NearestNeighbors

class PersonalRandomForestRecommender:

    def __init__(self, user, drop=True, verbose=True, n_neighbors=20,
                 positive_tracks_per_user=200, positive_portion=0.3):
        self.utm = userTrackMatrix(drop=drop, verbose=verbose)
        self.utm.load()
        self.user = user
        self.n_neighbors = n_neighbors
