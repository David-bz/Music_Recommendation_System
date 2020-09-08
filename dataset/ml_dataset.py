from dataset.user_item_matrix import userTrackMatrix
import pandas as pd
import numpy as np

class MLDataset:
    def __init__(self, verbose=True):
        self.verbose = verbose

        user_track_mat = userTrackMatrix(verbose=self.verbose)
        user_track_mat.load()
        user_track_mat.data.load_scaled_dataset()

        self.user_track_mat = user_track_mat
        self.init_dir = self.user_track_mat.init_dir
        self.n_components = self.user_track_mat.n_components
        self.orig_users = self.user_track_mat.users
        self.users = self.user_track_mat.data.scaled_users
        self.orig_tracks = self.user_track_mat.tracks
        self.tracks = self.user_track_mat.data.scaled_tracks
        self.loved = self.user_track_mat.loved
        self.played = self.user_track_mat.played
        self.MF_users = self.user_track_mat.MF_users
        self.MF_tracks = self.user_track_mat.MF_tracks
        self.data = pd.DataFrame(columns=self.get_ml_dataset_columns())

    def get_ml_dataset_columns(self):
        columns = []

        for i in range(self.n_components):
            columns.append('MF_users_{}'.format(i))
        for i in range(self.n_components):
            columns.append('MF_tracks_{}'.format(i))
        for col in self.users.columns:
            columns.append(col)
        for col in self.tracks.columns:
            columns.append(col)
        columns.append('score')

        return columns

    def add_user_track(self, user, track):
        row = self.MF_users[user, :].tolist() + self.MF_tracks[:, track].tolist()
        row = row + self.users.iloc[user].tolist() + self.tracks.iloc[track].tolist()
        row = row + [self.user_track_mat.mat[user, track]]
        self.data.loc[len(self.data)] = row

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

    def dump(self, path='ml_dataset.csv.zip'):
        pd.DataFrame.to_csv(self.data, self.init_dir + path, index=False, header=True, compression='zip')

    def load(self, path='ml_dataset.csv.zip'):
        self.loved = pd.read_csv(self.init_dir + path, header=0, compression='zip')

    def build_dataset(self, positive_portion=0.3, samples_mum=100000, dump=True):
        samples = self.get_positive_samples(int(samples_mum * positive_portion))
        samples += self.get_non_positive_samples(samples_mum - len(samples))
        np.random.shuffle(samples)

        if self.verbose:
            print('adding {} user-tracks pairs to dataset'.format(len(samples)))
        for user,track in samples:
            self.add_user_track(user, track)
            if self.verbose and len(self.data) % 100 == 0:
                print('{} user-tracks pairs already added to dataset'.format(len(self.data)))

        if dump:
            self.dump()


if __name__ == '__main__':
    dataset = MLDataset()
    dataset.build_dataset()
    print(0)
