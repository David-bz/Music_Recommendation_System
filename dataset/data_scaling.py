import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer, KBinsDiscretizer, MinMaxScaler
from time import gmtime
import pickle

class Dataset:
    def __init__(self, verbose = False):
        self.tracks =  pd.read_csv('./dataset/entities/tracks.csv.zip', compression='zip')
        self.users = pd.read_csv('./dataset/entities/users.csv.zip', index_col=0, compression='zip')
        self.scalers = {'users' : {}, 'tracks' : {}}
        self.verbose = verbose
        np.random.seed(1)

    def scale(self, save=True, verbose=False):
        if verbose: self.verbose = True
        self.scale_users()
        self.scale_tracks()
        if save:
            self.save_dataset()

    def save_dataset(self):
        pd.DataFrame.to_csv(self.users, './dataset/entities/scaled_users.csv.zip', index=False, header=True, compression='zip')
        pd.DataFrame.to_csv(self.tracks, './dataset/entities/scaled_tracks.csv.zip', index=False, header=True, compression='zip')
        with open('./dataset/entities/scalers.pickle', 'wb+') as handler:
            pickle.dump(self.scalers, handler, protocol=pickle.HIGHEST_PROTOCOL)



    def plot_and_save(self, data, feature_name, bins = 50, period='before'):
        hist, bins, _ = plt.hist(data, bins=bins)
        plt.xlabel(feature_name)
        plt.savefig('./dataset/distributions/users/' + feature_name + '_' + period + '_scale.png')
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
        self.users['playcount'] = playcount
        self.scalers['users']['playcount'] = power_scaler

    def scale_users_loved_tracks(self):
        # consider split to zeros and the rest
        data = self.users['loved_tracks'].values.reshape(-1, 1)
        if self.verbose: self.plot_and_save(data, 'loved_tracks')
        power_scaler = PowerTransformer(method='yeo-johnson').fit(data)
        loved_tracks = power_scaler.transform(data)
        if self.verbose: self.plot_and_save(loved_tracks, 'loved_tracks', period='after')
        self.users['loved_tracks'] = loved_tracks
        self.scalers['users']['loved_tracks'] = power_scaler

    def scale_users_artists_count(self):
        data = self.users['artists_count'].values.reshape(-1, 1)
        if self.verbose: self.plot_and_save(data, 'artists_count')
        power_scaler = PowerTransformer(method='box-cox').fit(data)
        artists_count = power_scaler.transform(data)
        if self.verbose: self.plot_and_save(artists_count, 'artists_count', period='after')
        self.users['artists_count'] = artists_count
        self.scalers['users']['artists_count'] = power_scaler

    def scale_registration_year(self):
        # data = self.users['registration_unix_time'].apply(lambda year : gmtime(year).tm_year).values.reshape(-1, 1)
        data = self.users['registration_unix_time'].values.reshape(-1, 1)
        if self.verbose:  self.plot_and_save(data, 'registration_unix_time', bins = 100)
        # power_scaler = PowerTransformer(method='box-cox').fit(data)
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data)
        reg_year = scaler.transform(data)
        if self.verbose:  self.plot_and_save(reg_year , 'registration_unix_time', period='after', bins = 100)
        self.users['registration_unix_time'] = reg_year
        self.scalers['users']['registration_unix_time'] = scaler

    def scale_last_activity_year(self):
        # consider split to zeros and the rest - here we should probably split to `still_active` and the rest
        indices_to_impute = self.users[self.users.last_activity_unix_time < 10000000].index.values
        avg_activity = self.users.activity_period_unix_time.mean()
        for i in indices_to_impute:
            self.users.at[i, 'last_activity_unix_time'] = self.users.iloc[i]['registration_unix_time'] + avg_activity

        # data = self.users['last_activity_unix_time'].apply(lambda year : gmtime(year).tm_year).values.reshape(-1, 1)
        data = self.users['last_activity_unix_time'].values.reshape(-1, 1)
        if self.verbose:  self.plot_and_save(data, 'last_activity_unix_time', bins = 100)
        # power_scaler = PowerTransformer(method='box-cox').fit(data)
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data)
        last_year = scaler.transform(data)
        if self.verbose:  self.plot_and_save(last_year  , 'last_activity_unix_time', period='after', bins = 100)
        self.users['last_activity_unix_time'] = last_year
        self.scalers['users']['last_activity_unix_time'] = scaler

    def scale_activity_period_days(self):
        self.users['activity_period_days'] = ((self.users['last_activity_unix_time'] - self.users['registration_unix_time'])/(60*60*24))
        data = self.users['activity_period_days'].values.reshape(-1 ,1)
        if self.verbose:  self.plot_and_save(data, 'activity_period_days')
        scaler = MinMaxScaler().fit(data)
        activity_in_days = scaler.transform(data)
        if self.verbose:  self.plot_and_save(activity_in_days, 'activity_period_days', period='after')
        self.users['activity_period_days'] = activity_in_days
        self.users.drop('activity_period_unix_time', axis = 1) # Unnecessary due to scaling
        self.scalers['users']['activity_period_days'] = scaler

    def scale_tracks(self):
        self.scale_track_duration()
        self.scale_track_listeners()
        self.scale_track_playcount()
        self.scale_track_artist_listeners()
        self.scale_track_artist_playcount()
        self.scale_track_album_listeners()
        self.scale_track_album_playcount()

    def impute_from_feature_distribution(self, feature, missing_value):
        missing = self.tracks[feature] == missing_value
        not_missing = self.tracks[feature][missing == False]
        std = not_missing.std()
        mean = not_missing.mean()
        self.tracks[feature] = self.tracks[feature].apply(
            lambda val: val if val != missing_value else np.random.normal(mean, std, 1)[0])

    def perform_log_scale(self, feature):
        power_scaler = PowerTransformer(method='yeo-johnson')
        data = self.tracks[feature].values.reshape(-1, 1)
        power_scaler = power_scaler.fit(data)
        data = power_scaler.transform(data)
        if self.verbose:
            hist, bins, _ = plt.hist(data, bins=50)
            plt.savefig('./dataset/distributions/tracks/' + feature + '_after_scale.png')
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
        self.impute_from_feature_distribution('track_duration', 0)
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
                plt.savefig('./dataset/distributions/' + table_name + '/' + feature + '_before_scale.png')
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
    data.scale(verbose=True)



