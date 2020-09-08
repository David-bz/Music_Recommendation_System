import numpy as np
import pandas as pd
from dataset.data_scaling import *
import scipy.sparse as sps
from sklearn.decomposition import *
import os

class userTrackMatrix:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data = Dataset()
        self.users = self.data.users
        self.tracks = self.data.tracks
        self.shape = (len(self.users),len(self.tracks))
        self.mat =  sps.lil_matrix(self.shape)
        self.init_dir = get_working_dir() + '/dataset/'
        self.loved = pd.read_csv(self.init_dir + 'relations/loved.csv.zip', header=0, compression='zip')
        self.played = pd.read_csv(self.init_dir +'relations/recently_played.csv.zip', header=0, compression='zip')

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
        if self.verbose:
            print('sparsity {}'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1])))

    def process_played_events(self, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        self.process_table_events(self.played, score_fun)
        if self.verbose:
            print('sparsity {}'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1])))

    def dump(self, npz_path='relations/user_track_matrix.npz',
             users_path='relations/mf_users.npy',
             tracks_path='relations/mf_tracks.npy'):
        coo_matrix = self.mat.tocoo()
        sps.save_npz(self.init_dir +npz_path, coo_matrix)
        if self.n_components > 0:
            np.save(self.init_dir + users_path, self.MF_users)
            np.save(self.init_dir + tracks_path, self.MF_tracks)

    def load(self, npz_path='relations/user_track_matrix.npz',
             users_path='relations/mf_users.npy',
             tracks_path='relations/mf_tracks.npy'):
        self.mat = sps.load_npz(self.init_dir + npz_path).tolil()
        try:
            self.MF_tracks = np.load(self.init_dir + tracks_path)
            self.MF_users = np.load(self.init_dir + users_path)
            self.n_components = len(self.MF_users[0])
            if self.verbose:
                print('sparsity {}, {} components'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1]), self.n_components))
        except Exception as e:
            print(e)

    def matrix_factorize(self, n_components=8, init='nndsvd', l1_ratio = 0.5, tol=1e-4):
        self.n_components = n_components
        model = NMF(n_components=n_components, init=init, random_state=0,
                    verbose=self.verbose, max_iter=400, l1_ratio=l1_ratio,
                    tol=tol)
        self.MF_users = model.fit_transform(self.mat)
        self.MF_tracks = model.components_
        if self.verbose:
            print('{} components reconstruction error {}'.format(n_components, model.reconstruction_err_))
            print('sparsity {}'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1])))

    def get_mf_score(self, user, track):
        return sum([self.MF_users[user, i]*self.MF_tracks[i, track] for i in range(self.n_components)])

    def get_user_loved_tracks(self, user):
        # related_tracks contains all the tracks that this user loved
        return list(self.loved[self.loved.user_id == user].track_id.values)

    def get_user_played_tracks(self, user):
        # related_tracks contains all the tracks that this user played
        return list(self.played[self.played.user_id == user].track_id.values)

    def get_user_related_tracks(self, user):
        # related_tracks contains all the tracks that this user loved or played
        return self.get_user_loved_tracks(user) + self.get_user_played_tracks(user)

    def evaluate_mf(self, samples = 200):
        count = 0
        sum = 0
        for user in np.random.choice(self.shape[0], samples):
            # score[i] is the score of track_id == i
            scores = np.dot(self.MF_users[user, :], self.MF_tracks)
            # positions[i] is the i'th worst track by this user
            positions = np.argsort(scores)
            # related_tracks contains all the tracks that this user loved / played
            related_tracks = self.get_user_related_tracks(user)

            # calculate the average positions of the related tracks
            avg_position = 0
            related_tracks_len = len(related_tracks)
            for track in related_tracks:
                avg_position += (np.argwhere(positions == track)[0][0] / related_tracks_len)
            # the score of this user is average positions divide the number of tracks
            # so if avg_position is 500K, divide it by 609K the score is ~0.82
            # so if avg_position is 300K, divide it by 609K the score is ~0.49
            # remember that `positions` is sorted from the worst to the best track
            user_score = avg_position / self.shape[1]

            if self.verbose:
                print('user {}, score {}, avg position {}, {} related tracks'.format(
                    user, user_score, int(avg_position), len(related_tracks)))
            sum += user_score
            count +=1

        print('score {}'.format(sum / count))




if __name__ == '__main__':
    user_track_matrix = userTrackMatrix()

    # create a trivial score function - for each event increase the score by 1
    trivial = lambda object, i, j: object.mat[i, j] + 1

    user_track_matrix.process_loved_events(trivial)
    user_track_matrix.process_played_events(trivial)
    user_track_matrix.matrix_factorize()
    user_track_matrix.evaluate_mf()
    user_track_matrix.dump()
