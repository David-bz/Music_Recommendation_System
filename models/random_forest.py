from sklearn.ensemble import RandomForestRegressor
from models.evaluate import Evaluate
from dataset.ml_dataset import *
import numpy as np

class BaseEstimator:
    def __init__(self, verbose=True):
        self.verbose = verbose
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
        self.estimator = RandomForestRegressor(n_estimators=100, verbose=verbose, n_jobs=-1)
        self.estimator = self.estimator.fit(self.X, self.Y)
        np.random.seed(1)


    def predict(self, user):
        user_matedata = self.ml_data.get_user_row_part(user)
        track_ids = [track for track in range(self.tracks_num)]
        chunks = np.array_split(track_ids ,50)

        scores = []
        for i,chunk in enumerate(chunks):
            X_chuck = []
            for track in chunk:
                X_chuck.append(user_matedata + self.ml_data.get_track_row_part(track))

            scores += self.estimator.predict(X_chuck).tolist()
            if self.verbose:
                print('user {} {}%: {} tracks predicted'.format(user, 2*i, len(scores)))

        scores = np.argsort(scores)[::-1]
        return scores


if __name__ == '__main__':
    est = RandomForestRecommender()
    evaluator = Evaluate(est, 'simple_random_forest', K=250, samples=5)
    evaluator.evaluate()
