from dataset.ml_dataset import *
from models.vanilla_models import *
import numpy as np

class Evaluate:
    def __init__(self, model, K=100, verbose=True):
        " model is object with function predict:user -> list of tracks ids where the first track is the most recommended "
        self.model = model
        self.K = K
        self.verbose = verbose
        self.data = MLDataset()
        self.data.load()

    def evaluate_precision_at_k(self, samples):
        results = []
        user_ids = np.random.choice(self.data.user_track_mat.shape[0], samples, replace=False)
        if self.verbose:
            print('calculate evaluate precision at k for {} users'.format(samples))

        for user in user_ids:
            related_tracks = self.data.user_track_mat.get_user_related_tracks(user)
            recommended = self.model.predict(user)[:self.K]
            assert len(recommended) > 0

            success = sum([1 if track in related_tracks else 0 for track in recommended])
            results.append(success / len(recommended))
        return np.average(results)

    def evaluate_recall_at_k(self, samples):
        results = []
        user_ids = np.random.choice(self.data.user_track_mat.shape[0], samples, replace=False)
        if self.verbose:
            print('calculate evaluate recall at k for {} users'.format(samples))

        for user in user_ids:
            related_tracks = self.data.user_track_mat.get_user_related_tracks(user)
            recommended = self.model.predict(user)[:self.K]
            assert len(recommended) > 0

            success = sum([1 if track in recommended else 0 for track in related_tracks])
            results.append(success / min(len(related_tracks), self.K))
        return np.average(results)

    def get_position(self, track, recommended):
        try:
            pos = recommended.tolist().index(track)
        except ValueError:
            pos = self.data.user_track_mat.shape[1]
        return pos

    def evaluate_map(self, samples):
        results = []
        user_ids = np.random.choice(self.data.user_track_mat.shape[0], samples, replace=False)
        if self.verbose:
            print('calculate map for {} users'.format(samples))

        for user in user_ids:
            loved_tracks = self.data.user_track_mat.get_user_loved_tracks(user)
            played_tracks = self.data.user_track_mat.get_user_played_tracks(user)
            related_tracks = played_tracks + loved_tracks
            recommended = self.model.predict(user)

            loved_positions = [self.get_position(track, recommended) for track in loved_tracks]
            played_positions = [self.get_position(track, recommended) for track in played_tracks]
            related_positions = [self.get_position(track, recommended) for track in related_tracks]

            result = (np.average(loved_positions), np.average(played_positions), np.average(related_positions))
            results.append(result)
            print('evaluate_map after user {}, results (loved, played, related) : {}'.format(user, results))
        return np.average(results, 0)

    def evaluate(self, samples=200):
        precision_at_k = self.evaluate_precision_at_k(samples)
        if self.verbose:
            print('precision at k {}'.format(precision_at_k))

        recall_at_k = self.evaluate_recall_at_k(samples)
        if self.verbose:
            print('recall at k {}'.format(recall_at_k))

        map = self.evaluate_map(samples)
        if self.verbose:
            print('map {}'.format(map))

        return (precision_at_k, recall_at_k, map)

if __name__ == "__main__":
    vanilla_mf = Vanilla_MF()
    evaluator = Evaluate(vanilla_mf, K=250)
    evaluator.evaluate(50)


