from dataset.user_item_matrix import userTrackMatrix
import pandas as pd
import numpy as np
from utils import generate_path
from sklearn.neighbors import NearestNeighbors


class MLDataset:
    def __init__(self, verbose=True, drop=False):
        self.verbose = verbose

        user_track_mat = userTrackMatrix(verbose=self.verbose, drop=drop)
        user_track_mat.load()
        user_track_mat.data.load_scaled_dataset()

        self.user_track_mat = user_track_mat
        self.n_components = self.user_track_mat.n_components
        self.orig_users = self.user_track_mat.users
        self.users = self.user_track_mat.data.scaled_users
        self.orig_tracks = self.user_track_mat.tracks
        self.tracks = self.user_track_mat.data.scaled_tracks
        self.MF_users = self.user_track_mat.MF_users
        self.MF_tracks = self.user_track_mat.MF_tracks
        self.data = pd.DataFrame(columns=self.get_ml_dataset_columns())

    def reset_data(self):
        self.data = pd.DataFrame(columns=self.get_ml_dataset_columns())

    def get_ml_dataset_columns(self):
        columns = []

        for i in range(self.n_components):
            columns.append('MF_users_{}'.format(i))
        for col in self.users.columns:
            columns.append(col)
        for i in range(self.n_components):
            columns.append('MF_tracks_{}'.format(i))
        for col in self.tracks.columns:
            columns.append(col)
        columns.append('score')

        return columns

    def get_user_row_part(self, user):
        row = self.MF_users[user, :].tolist()
        row += self.users.iloc[user].tolist()
        return row

    def get_track_row_part(self, track):
        row = self.MF_tracks[:, track].tolist()
        row += self.tracks.iloc[track].tolist()
        return row

    def get_user_track_score(self, user, track):
        return [self.user_track_mat.mat[user, track]]

    def get_user_track_row(self, user, track):
        return self.get_user_row_part(user) + self.get_track_row_part(track) + self.get_user_track_score(user, track)

    def add_user_track(self, user, track):
        self.data.loc[len(self.data)] = self.get_user_track_row(user, track)

    def get_positive_samples(self, samples_num):
        samples = []
        user_ids = [i for i in range(self.user_track_mat.shape[0])]
        np.random.shuffle(user_ids)

        if self.verbose:
            if samples_num < len(user_ids)*5:
                print("{} users won't be covered in the dataset".format(int(len(user_ids) - (samples_num / 5))))
            print('collecting {} positive tracks'.format(samples_num))

        while len(samples) < samples_num:
            if len(user_ids) > 0:
                user = list.pop(user_ids)
            else:
                user = np.random.choice(self.user_track_mat.shape[0], 1)[0]
            user_related_tracks = self.user_track_mat.get_user_related_tracks(user)
            tracks = np.random.choice(user_related_tracks, min(len(user_related_tracks), 5))
            for track in tracks:
                samples.append((user, track))
        return samples

    def get_non_positive_samples(self, samples_num):
        samples = []
        user_ids = [i for i in range(self.user_track_mat.shape[0])]
        np.random.shuffle(user_ids)

        if self.verbose:
            if samples_num < len(user_ids)*5:
                print("{} users won't be covered in the dataset".format(int(len(user_ids) - samples_num / 5)))
            print('collecting {} non-positive tracks'.format(samples_num))

        while len(samples) < samples_num:
            if len(user_ids) > 0:
                user = list.pop(user_ids)
            else:
                user = np.random.choice(self.user_track_mat.shape[0], 1)[0]
            related_tracks = self.user_track_mat.get_user_related_tracks(user)
            tracks = np.random.choice(self.user_track_mat.shape[1],  5)
            for track in tracks:
                if track not in related_tracks:
                    samples.append((user, track))
        return samples

    def dump(self, path='dataset/ml_dataset.csv.zip'):
        pd.DataFrame.to_csv(self.data, generate_path(path), index=False, header=True, compression='zip')

    def load(self, path='dataset/ml_dataset.csv.zip'):
        self.data = pd.read_csv(generate_path(path), header=0, compression='zip')

    def populate_samples(self, samples):
        for user, track in samples:
            self.add_user_track(user, track)
            if self.verbose and len(self.data) % 500 == 0:
                print('{} out of {} user-tracks pairs already added to dataset'.format(len(self.data), len(samples)))

    def build_dataset(self, positive_portion=0.3, samples_mum=100000, dump=True):
        samples = self.get_positive_samples(int(samples_mum * positive_portion))
        samples += self.get_non_positive_samples(samples_mum - len(samples))
        np.random.shuffle(samples)

        if self.verbose:
            print('adding {} user-tracks pairs to dataset'.format(len(samples)))
        self.populate_samples(samples)

        if dump:
            self.dump()

    def get_nearest_users(self, user, n_neighbors):
        NN = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree')
        NN.fit(self.user_track_mat.MF_users)
        user_vector = self.user_track_mat.MF_users[user].reshape(1, -1)
        distance, index = NN.kneighbors(user_vector)
        nearest_user_idx = list(index[0, 1:])
        return nearest_user_idx

    def build_dataset_for_user(self, user, n_neighbors, positive_tracks_per_neighbor=200, positive_portion=0.3):
        samples = []

        # collect all the positive tracks for the pivot user
        if self.verbose:
            print('collect all the positive tracks for the pivot user {}'.format(user))
        pivot_user_related_tracks = self.user_track_mat.get_user_related_tracks(user)
        for track in pivot_user_related_tracks:
            samples.append((user, track))

        # collect non positive tracks for the pivot user
        non_positive_tracks_num = (1-positive_portion) * len(samples) / positive_portion
        if self.verbose:
            print('collect {} non positive tracks for the pivot user {}'.format(non_positive_tracks_num, user))
        non_positive_tracks_count = 0
        while non_positive_tracks_count < non_positive_tracks_num:
            track = np.random.choice(self.user_track_mat.shape[1], 1)[0]
            if track not in pivot_user_related_tracks:
                samples.append((user, track))
                non_positive_tracks_count += 1

        # get n_neighbors nearest neighbors of this pivot user
        if self.verbose:
            print('collect {} nearest neighbors of user {}'.format(n_neighbors, user))
        neighbors = self.get_nearest_users(user, n_neighbors)

        for i, neighbor in enumerate(neighbors):
            if self.verbose:
                print('collect neighbor {} ({} out of {})'.format(neighbor, i, n_neighbors))

            # collect up to positive_tracks_per_neighbor for this neighbor
            all_neighbor_positive_tracks = self.user_track_mat.get_user_related_tracks(neighbor)
            neighbor_positive_tracks = all_neighbor_positive_tracks[:positive_tracks_per_neighbor]
            for track in neighbor_positive_tracks:
                samples.append((neighbor, track))

            # collect non positive tracks for this neighbor
            non_positive_tracks_num = (1-positive_portion) * positive_tracks_per_neighbor / positive_portion
            non_positive_tracks_count = 0
            while non_positive_tracks_count < non_positive_tracks_num:
                track = np.random.choice(self.user_track_mat.shape[1], 1)[0]
                if track not in all_neighbor_positive_tracks:
                    samples.append((neighbor, track))
                    non_positive_tracks_count += 1

        if self.verbose:
            print('{} user-track pairs of {} nearest neighbors collected'.format(len(samples), n_neighbors))
        self.populate_samples(samples)


if __name__ == '__main__':
    dataset = MLDataset()
    dataset.build_dataset()
    print(0)
