from dataset.ml_dataset import *
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time

class BaseVanilla:
    def __init__(self, drop=False, MF_users=None, MF_tracks=None):
        self.user_track_matrix = userTrackMatrix(drop)
        self.user_track_matrix.load()

        if MF_users is not None:
            self.MF_users = MF_users
        else:
            self.MF_users = self.user_track_matrix.MF_users

        if MF_tracks is not None:
            self.MF_tracks = MF_tracks
        else:
            self.MF_tracks = self.user_track_matrix.MF_tracks

class Vanilla_MF(BaseVanilla):
    def __init__(self, MF_users=None, MF_tracks=None):
        super().__init__(MF_users=MF_users, MF_tracks=MF_tracks)

    def predict(self, user):
        scores = np.dot(self.MF_users[user, :], self.MF_tracks)
        positions = np.argsort(scores)[::-1]
        return positions

class VanillaNearestUsers(BaseVanilla):
    def __init__(self, K = 1):
        super().__init__()
        self.K = K
        self.NN = NearestNeighbors(n_neighbors=self.K + 1, algorithm='ball_tree', n_jobs=-1)
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


class VanillaNearestTracks(BaseVanilla):
    def __init__(self, n_neighbors = 3, tracks_limit = None, drop=False):
        super().__init__(drop)
        self.n_neighbors = n_neighbors
        self.tracks_limit = tracks_limit
        self.NN = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='brute', n_jobs=-1)
        print("starting to fit neighborhood...")
        st = time.time()
        self.NN.fit(self.MF_tracks.transpose())
        print("> finish fitting model: {:.3} secs".format(time.time() - st))

    def get_nearest_tracks(self, track_id):
        """ gets a track and returns its K most nearest tracks in the latent space """
        track_vector = self.MF_tracks[:, track_id].reshape(1, -1)
        distance, index = self.NN.kneighbors(track_vector)
        nearest_track_idx = list(index[0, 1:])
        return nearest_track_idx

    def predict(self, user):
        tracks_indices = self.user_track_matrix.mat.rows[user]
        tracks_scores = self.user_track_matrix.mat.data[user]
        # sort the indices by the best score
        tracks_indices = [idx for _,idx in sorted(zip(tracks_scores, tracks_indices), reverse = True)]
        recommend = []
        limit = len(tracks_indices) if self.tracks_limit == None else self.tracks_limit
        for track_id in tracks_indices[:limit]:
            a = time.time()
            print("getting nearest tracks")
            similar_tracks =  self.get_nearest_tracks(track_id)
            for t in similar_tracks:
                if t not in recommend: recommend.append(t)
            print("> got nearest tracks: {:.3} secs".format(time.time() - a))
        return recommend



if __name__ == '__main__':
    v = VanillaNearestTracks(3)
    print(v.predict(4000))








