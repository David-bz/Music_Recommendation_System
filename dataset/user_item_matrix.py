import numpy as np
import pandas as pd
from dataset.data_scaling import Dataset
import scipy.sparse as sps
from sklearn.decomposition import *

class userTrackMatrix:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data = Dataset()
        self.users = self.data.users
        self.tracks = self.data.tracks
        self.shape = (len(self.users),len(self.tracks))
        self.mat =  sps.lil_matrix(self.shape)
        self.loved = pd.read_csv('./dataset/relations/loved.csv.zip', header=0, compression='zip')
        self.played = pd.read_csv('./dataset/relations/recently_played.csv.zip', header=0, compression='zip')
        self.n_components = 0 # symbolise that both self.MF_users & self.MF_tracks didn't set yet

        if self.verbose:
            print('create users-tracks matrix {}'.format(self.shape))
            print('{} played events and {} loved events to process'.format(len(self.played), len(self.loved)))

    def get_item_count(self, table, attribute, val):
        return len(table[table[attribute] == val])

    def get_user_play_count(self, user_id):
        return self.get_item_count(self.played, 'user_id', user_id)

    def get_user_love_count(self, user_id):
        return self.get_item_count(self.loved, 'user_id', user_id)

    def get_track_play_count(self, track_id):
        return self.get_item_count(self.played, 'track_id', track_id)

    def get_track_love_count(self, track_id):
        return self.get_item_count(self.loved, 'track_id', track_id)

    def process_table_events(self,table, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        if self.verbose:
            bucket_size = len(table) // 100

        for index, row in table.iterrows():
            self.mat[row.user_id, row.track_id] = score_fun(self, row.user_id, row.track_id)
            if self.verbose and index % bucket_size == 0:
                print('done {}%: {} out of {}'.format(index // bucket_size, index, len(table)))

    def process_loved_events(self, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        self.process_table_events(self.loved, score_fun)

    def process_played_events(self, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        self.process_table_events(self.played, score_fun)

    def dump(self, npz_path='./dataset/relations/user_track_matrix.npz',
             users_path='./dataset/relations/mf_users.npy',
             tracks_path='./dataset/relations/mf_tracks.npy'):
        coo_matrix = self.mat.tocoo()
        sps.save_npz(npz_path, coo_matrix)
        if self.n_components > 0:
            np.save(users_path, self.MF_users)
            np.save(tracks_path, self.MF_tracks)

    def load(self, npz_path='./dataset/relations/user_track_matrix.npz',
             users_path='./dataset/relations/mf_users.npy',
             tracks_path='./dataset/relations/mf_tracks.npy'):
        self.mat = sps.load_npz(npz_path).tolil()
        try:
            self.MF_tracks = np.load(tracks_path)
            self.MF_users = np.load(users_path)
            self.n_components = len(self.MF_users[0])
        except Exception as e:
            print(e)

    def matrix_factorize(self, n_components=8):
        self.n_components = n_components
        model = NMF(n_components=n_components, init='random', random_state=0, verbose=self.verbose, max_iter=400)
        self.MF_users = model.fit_transform(self.mat)
        self.MF_tracks = model.components_
        if self.verbose:
            print('{} components reconstruction error {}'.format(n_components, model.reconstruction_err_))

    def get_mf_score(self, user, track):
        return sum([self.MF_users[user, i]*self.MF_tracks[i, track] for i in range(self.n_components)])

if __name__ == '__main__':
    user_track_matrix = userTrackMatrix()

    # create a trivial score function - for each event increase the score by 1
    trivial = lambda object, i, j: object.mat[i, j] + 1

    user_track_matrix.process_loved_events(trivial)
    user_track_matrix.process_played_events(trivial)
    user_track_matrix.matrix_factorize()
    user_track_matrix.dump()
    print(0)