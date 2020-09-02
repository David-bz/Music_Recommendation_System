import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer, KBinsDiscretizer
from time import gmtime

class Dataset:
    def __init__(self, verbose = False):
        self.tracks =  pd.read_csv('./entities/tracks.csv.zip', compression='zip')
        self.users = pd.read_csv('./entities/users.csv.zip', index_col=0, compression='zip')
        self.scalers = {'users' : {}, 'tracks' : {}}
        self.verbose = verbose

    def plot_and_save(self, data, feature_name, bins = 50, period='before'):
        hist, bins, _ = plt.hist(data, bins=bins)
        plt.xlabel(feature_name)
        plt.savefig('./distributions/users/' + feature_name + '/' + period + '_scale.png')
        plt.show()

    def scale_users(self, verbose = False):
        if verbose:
            self.verbose = True
        self.scale_users_playcount()
        self.scale_users_loved_tracks()
        self.scale_users_artists_count()
        self.scale_registration_year()
        self.scale_last_activity_year()
        self.scale_activity_period_days()

    def scale_users_playcount(self, verbose=False):
        # Avoid the case when users.playcount < users.loved_tracks (match to the loved_tracks)
        self.users['playcount'] = self.users.apply(lambda row : row['loved_tracks'] if row['playcount'] < row['loved_tracks'] else row['playcount'], axis=1)
        data = self.users['playcount'].values.reshape(-1, 1)
        if self.verbose: self.plot_and_save(data, 'playcount')
        power_scaler = PowerTransformer(method='box-cox').fit(data)
        playcount = power_scaler.transform(data)
        if self.verbose: self.plot_and_save(playcount, 'playcount', period='after')

        self.scalers['users']['playcount'] = power_scaler

    def scale_users_loved_tracks(self):
        data = self.users['loved_tracks'].values.reshape(-1, 1)
        if self.verbose: self.plot_and_save(data, 'loved_tracks')
        power_scaler = PowerTransformer(method='yeo-johnson').fit(data)
        loved_tracks = power_scaler.transform(data)
        if self.verbose: self.plot_and_save(loved_tracks, 'loved_tracks', period='after')

    def scale_users_artists_count(self):
        data = self.users['artists_count'].values.reshape(-1, 1)
        if self.verbose: self.plot_and_save(data, 'artists_count')
        power_scaler = PowerTransformer(method='box-cox').fit(data)
        artists_count = power_scaler.transform(data)
        if self.verbose: self.plot_and_save(artists_count, 'artists_count', period='after')

    def scale_registration_year(self):
        # data = self.users['registration_unix_time'].apply(lambda year : gmtime(year).tm_year).values.reshape(-1, 1)
        data = self.users['registration_unix_time'].values.reshape(-1, 1)
        if self.verbose:  self.plot_and_save(data, 'registration_unix_time', bins = 20)

    def scale_last_activity_year(self):
        indices_to_impute = self.users[self.users.last_activity_unix_time < 10000000].index.values
        avg_activity = self.users.activity_period_unix_time.mean()
        for i in indices_to_impute:
            self.users.at[i, 'last_activity_unix_time'] = self.users.iloc[i]['registration_unix_time'] + avg_activity

        # data = self.users['last_activity_unix_time'].apply(lambda year : gmtime(year).tm_year).values.reshape(-1, 1)
        data = self.users['last_activity_unix_time'].values.reshape(-1, 1)
        if self.verbose:  self.plot_and_save(data, 'last_activity_unix_time', bins = 20)

    def scale_activity_period_days(self):
        self.users['activity_period_days'] = ((self.users['last_activity_unix_time'] - self.users['registration_unix_time'])/(60*60*24)).astype(int)
        data = self.users['activity_period_days']
        if self.verbose:  self.plot_and_save(data, 'activity_period_days')

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
        return

    def tracks_split_zeros(self, feature, discrete_to_bins=True, bins=4):
        zeros = self.tracks[feature] == 0
        self.tracks['{}_zeros'.format(feature)] = zeros

        if not discrete_to_bins:
            return

        not_zeros = self.tracks[feature][zeros == False].values.reshape(-1, 1)
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        est = est.fit(not_zeros)

        for bin in range(bins):
            bigger_than = self.tracks[feature] >= est.bin_edges_[0][bin]

            if bin != bins - 1:
                small_than = self.tracks[feature] < est.bin_edges_[0][bin + 1]
            else:
                small_than = self.tracks[feature] <= est.bin_edges_[0][bin + 1]

            this_bin = bigger_than & small_than
            self.tracks['{}bin{}'.format(feature, bin)] = this_bin

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
        missing = (self.tracks['album_listeners'] == -1).astype(int)
        self.tracks['is_single'] = missing
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
    data.scale_users(True)



