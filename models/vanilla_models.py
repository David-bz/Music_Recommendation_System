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

    def get_nearest_users(self, user_id):
        """ finds for a given user the most similar user, and returns its songs as predictions """
        user_vector = self.MF_users[user_id].reshape(1, -1)
        distance, index = self.NN.kneighbors(user_vector)
        nearest_user_idx = list(index[0, 1:])
        return nearest_user_idx

    def predict(self, user):
        nearest_user_idx = self.get_nearest_users(user)
        nearest = dict()
        for user_id in nearest_user_idx:
            idx_to_recommend =  self.user_track_matrix.mat.rows[user_id]
            scores_to_recommend = self.user_track_matrix.mat.data[user_id]
            assert len(idx_to_recommend) == len(scores_to_recommend)
            for idx, score in zip(idx_to_recommend, scores_to_recommend):
                nearest[idx] = nearest.get(idx, 0.) + score

        indices = [k for k, v in sorted(nearest.items(), key = lambda item : item[1], reverse = True)]
        return indices

if __name__ == '__main__':
    v = Vanilla_NN(3)
    print(v.predict(4000))








