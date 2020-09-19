from dataset.ml_dataset import *
from sklearn.ensemble import RandomForestRegressor
from models.evaluate import Evaluate


class PersonalRandomForestRecommender:

    def __init__(self, drop=True, verbose=True, n_neighbors=20,
                 positive_tracks_per_neighbor=200, positive_portion=0.3):
        self.drop = drop
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.positive_tracks_per_neighbor = positive_tracks_per_neighbor
        self.positive_portion = positive_portion
        self.ml_data = MLDataset(drop=drop, verbose=verbose)
        self.seed = 1
        np.random.seed(self.seed)
        self.tracks_num = self.ml_data.user_track_mat.shape[1]
        self.user = None
        self.est = None

    def train(self, user):
        self.user = user
        self.ml_data.reset_data()
        self.ml_data.build_dataset_for_user(user, self.n_neighbors,
                                            positive_tracks_per_neighbor=self.positive_tracks_per_neighbor,
                                            positive_portion=self.positive_portion)
        X = self.ml_data.data.drop(['score'], axis=1).values
        Y = self.ml_data.data['score'].values
        self.est = RandomForestRegressor(n_estimators=100, verbose=self.verbose, n_jobs=-1)
        self.est = self.est.fit(X, Y)

    def predict(self, user):
        if self.user is None or self.user != user:
            self.train(user)

        user_metadata = self.ml_data.get_user_row_part(user)
        track_ids = [track for track in range(self.tracks_num)]
        num_chunks = 25
        chunks = np.array_split(track_ids, num_chunks)
        scores = []

        for i, chunk in enumerate(chunks):
            X_chuck = []
            for track in chunk:
                X_chuck.append(user_metadata + self.ml_data.get_track_row_part(track))

            scores += self.est.predict(X_chuck).tolist()
            if self.verbose:
                print('user {} {}%: {} tracks predicted'.format(user, (100/num_chunks)*i, len(scores)))

        scores = np.argsort(scores)[::-1]
        return scores


if __name__ == '__main__':
    est = PersonalRandomForestRecommender(drop=True, verbose=True, n_neighbors=20,
                                          positive_tracks_per_neighbor=200, positive_portion=0.3)
    evaluator = Evaluate(est, 'personal_random_forest', K=250, samples=5)
    evaluator.evaluate()
