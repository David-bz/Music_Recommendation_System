import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer


class Dataset:
    def __init__(self):
        self.tracks =  pd.read_csv('./entities/tracks.csv.zip', compression='zip')
        self.users = pd.read_csv('./entities/users.csv.zip', index_col=0, compression='zip')
        self.scalers = {'users' : {}, 'tracks' : {}}

    def scale_users(self):
        self.scale_users_playcount()


    def scale_users_playcount(self, verbose=False):
        # Avoid the case when users.playcount < users.loved_tracks (match to the loved_tracks)
        self.users['playcount'] = self.users.apply(lambda row : row['loved_tracks'] if row['playcount'] < row['loved_tracks'] else row['playcount'], axis=1)
        power_scaler = PowerTransformer(method='box-cox')
        data = self.users['playcount'].values.reshape(-1, 1)
        power_scaler = power_scaler.fit(data)
        pc = power_scaler.transform(data)
        if verbose:
            hist, bins, _ = plt.hist(pc, bins=50)
            plt.savefig('./distributions/users/playcount/after_scale.png')
            plt.show()
        self.scalers['users']['playcount'] = power_scaler



    def scale_tracks(self):
        self.scale_track_duration()
        self.scale_track_listeners()
        self.scale_track_playcount()
        self.scale_track_artist_listeners()
        self.scale_track_artist_playcount()
        self.scale_track_album_listeners()
        self.scale_track_album_playcount()

    def perform_log_scale(self, feature):
        power_scaler = PowerTransformer(method='yeo-johnson')
        data = self.tracks[feature].values.reshape(-1, 1)
        power_scaler = power_scaler.fit(data)
        data = power_scaler.transform(data)
        if self.verbose:
            hist, bins, _ = plt.hist(data, bins=50)
            plt.savefig('./distributions/tracks/' + feature + '_after_scale.png')
            plt.show()
        self.scalers['tracks'][feature] = power_scaler
        self.tracks[feature] = data


    def scale_track_duration(self):
        # consider split to zeros and the rest
        self.perform_log_scale('track_duration')

    def scale_track_listeners(self):
        self.perform_log_scale('track_listeners')

    def scale_track_playcount(self):
        self.perform_log_scale('track_playcount')

    def scale_track_artist_listeners(self):
        self.perform_log_scale('artist_listeners')

    def scale_track_artist_playcount(self):
        self.perform_log_scale('artist_playcount')

    def scale_track_album_listeners(self):
        # consider split to zeros and the rest
        self.perform_log_scale('album_listeners')

    def scale_track_album_playcount(self):
        # consider split to zeros and the rest
        self.perform_log_scale('album_playcount')

    def show_table_distributions(self, table, table_name):
        for feature in table.columns:
            try:
                ax = table[feature].plot.hist(bins=50)
                plt.xlabel(feature)
                plt.savefig('./distributions/' + table_name + '/' + feature + '_before_scale.png')
                plt.show()
            except TypeError:
                print(feature)
                continue

    def show_users_distributions(self):
        self.show_table_distributions(self.users, 'users')

    def show_tracks_distributions(self):
        self.show_table_distributions(self.tracks, 'tracks')

if __name__ == '__main__':
    data = Dataset()
    data.show_tracks_distributions()
    data.scale_tracks()



