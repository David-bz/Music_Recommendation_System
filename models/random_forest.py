from sklearn.ensemble import RandomForestRegressor
from models.evaluate import Evaluate
from dataset.ml_dataset import *
import numpy as np

class BaseEstimator:
    def __init__(self, verbose=True):
        self.ml_data = MLDataset()
        self.ml_data.load()
        self.users_num = self.ml_data.user_track_mat.shape[0]
        self.tracks_num = self.ml_data.user_track_mat.shape[1]
        self.X = self.ml_data.data.drop(['score'], axis=1).values
        self.Y = self.ml_data.data['score'].values

    def predict(self, user):
        " this method should return list of tracks ids from the most to the least recommended "
        pass


class RandomForestRecommender(BaseEstimator):
    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)
        self.estimator = RandomForestRegressor(verbose=verbose, n_jobs=-1)
        self.estimator = self.estimator.fit(self.X, self.Y)

    def predict(self, user):
        scores = []
        for track in range(self.tracks_num):
            user_track_row = self.ml_data.get_user_track_row(user, track)[:-1]
            scores.append(self.estimator.predict(user_track_row)[0])

        scores = np.argsort(scores)[::-1]
        return scores


if __name__ == '__main__':
    est = RandomForestRecommender()
    evaluator = Evaluate(est)
    evaluator.evaluate(50)
