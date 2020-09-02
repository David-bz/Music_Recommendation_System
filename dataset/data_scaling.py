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
        pass

    def show_users_distributions(self):
        for feature in self.users.columns:
            try:
                ax = self.users[feature].plot.hist(bins=50)
                plt.xlabel(feature)
                plt.savefig('./users_distributions/' + feature + '.png')
                plt.show()
            except TypeError:
                print(feature)
                continue



if __name__ == '__main__':
    data = Dataset()
    data.scale_users()









