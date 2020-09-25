from dataset.ml_dataset import *
from models.vanilla_models import *
import numpy as np
import json

class Evaluate:
    def __init__(self, model, model_description, K=100, drop=False, random_samples=0, verbose=True):
        " model is object with function predict:user -> list of tracks ids where the first track is the most recommended "
        self.model = model
        self.model_description = model_description
        self.K = K
        self.random_samples = random_samples
        self.verbose = verbose
        self.drop = drop
        self.data = MLDataset(drop = self.drop)
        self.data.load()

    def add_results(self, user, results_type, loved_score, played_score, related_score, dropped_score = np.nan):
        results = {
            'loved': loved_score,
            'played_score': played_score,
            'related_score': related_score
        }
        if self.drop: results['dropped_score'] = dropped_score
        self.results['{}'.format(user)][results_type] = results
        if self.verbose:
            print('{}: {}\n'.format(results_type, results))

    def evaluate_precision_at_k(self, users):
        if self.verbose:
            print('calculate evaluate precision at k for {} users'.format(len(users)))

        for user, loved_tracks, played_tracks, related_tracks, dropped_tracks, recommended in users:
            recommended = recommended[:self.K]
            loved_score = sum([1 if track in loved_tracks else 0 for track in recommended]) / len(recommended)
            played_score = sum([1 if track in played_tracks else 0 for track in recommended]) / len(recommended)
            related_score = sum([1 if track in related_tracks else 0 for track in recommended]) / len(recommended)
            if self.drop and len(dropped_tracks) > 0:
                dropped_score = sum([1 if track in dropped_tracks else 0 for track in recommended]) / len(recommended)
            else:
                dropped_score = np.nan
            self.add_results(user, 'P@K' , loved_score, played_score, related_score, dropped_score)

    def evaluate_recall_at_k(self, users):
        if self.verbose:
            print('calculate evaluate recall at k for {} users'.format(len(users)))

        for user, loved_tracks, played_tracks, related_tracks, dropped_tracks, recommended in users:
            recommended = recommended[:self.K]
            if len(loved_tracks) == 0:
                loved_score = np.nan
            else:
                loved_score = sum([1 if track in recommended else 0 for track in loved_tracks]) / min(len(loved_tracks), self.K)

            if len(played_tracks) == 0:
                played_score = np.nan
            else:
                played_score = sum([1 if track in recommended else 0 for track in played_tracks]) / min(len(played_tracks), self.K)

            if len(related_tracks) == 0:
                related_score = np.nan
            else:
                related_score = sum([1 if track in recommended else 0 for track in related_tracks]) / min(len(related_tracks), self.K)

            if self.drop and len(dropped_tracks) > 0:
                dropped_score = sum([1 if track in recommended else 0 for track in dropped_tracks]) / min(len(dropped_tracks), self.K)
            else:
                dropped_score = np.nan

            self.add_results(user, 'R@K' , loved_score, played_score, related_score, dropped_score)

    def get_position(self, track, recommended):
        try:
            if not isinstance(recommended, list):
                recommended = recommended.tolist()
            pos = recommended.index(track)
        except ValueError:
            last_track = self.data.user_track_mat.shape[1]
            next_track = len(recommended)
            pos = (next_track + last_track) // 2
        return pos

    def evaluate_map(self, users):
        if self.verbose:
            print('calculate map for {} users'.format(len(users)))

        for user, loved_tracks, played_tracks, related_tracks, dropped_tracks, recommended in users:
            related_positions = []
            if len(loved_tracks) == 0:
                loved_score = np.nan
            else:
                loved_positions = [self.get_position(track, recommended) for track in loved_tracks]
                related_positions += loved_positions
                loved_score = np.average(loved_positions)

            if len(played_tracks) == 0:
                played_score = np.nan
            else:
                played_positions = [self.get_position(track, recommended) for track in played_tracks]
                related_positions += played_positions
                played_score = np.average(played_positions)

            if len(related_tracks) == 0:
                related_score = np.nan
            else:
                related_score = np.average(related_positions)

            if not self.drop or len(dropped_tracks) == 0:
                dropped_score = np.nan
            else:
                dropped_positions = [self.get_position(track, recommended) for track in dropped_tracks]
                dropped_score = np.average(dropped_positions)

            self.add_results(user, 'MAP', loved_score, played_score, related_score, dropped_score)


    def evaluate(self, input_users = None):
        if input_users == None and self.random_samples > 0:
            user_ids = np.random.choice(self.data.user_track_mat.shape[0], self.random_samples, replace=False)
        else:
            if isinstance(input_users, list):
                user_ids = input_users
            elif isinstance(input_users, int):
                user_ids = [input_users]
            else: raise TypeError
        self.results = dict()
        self.results['info'] = {
            'K': self.K,
            'model': self.model_description,
            'drop' : str(self.drop),
            'samples': len(user_ids),
        }

        users = []
        for i,user in enumerate(user_ids):
            loved_tracks = self.data.user_track_mat.get_user_loved_tracks(user)
            played_tracks = self.data.user_track_mat.get_user_played_tracks(user)
            related_tracks = played_tracks + loved_tracks
            dropped_tracks = self.data.user_track_mat.get_user_dropped_tracks(user) if self.drop else []
            recommended = self.model.predict(user)

            users.append((user, loved_tracks, played_tracks, related_tracks, dropped_tracks, recommended))
            user_dict = {'loved_count': len(loved_tracks),
                         'played_count': len(played_tracks),
                         'dropped_tracks': len(dropped_tracks),
                         'top250': recommended[:250]}
            print(len(dropped_tracks))
            self.results['{}'.format(user)] = user_dict
            if self.verbose:
                print('got recommendations for {} users out of {}'.format(i+1, self.random_samples))

        self.evaluate_precision_at_k(users)
        self.evaluate_recall_at_k(users)
        self.evaluate_map(users)

        with open(generate_path('models/model_results/' + self.model_description + '.json'), 'w') as fp:
            json.dump(self.results, fp)


if __name__ == "__main__":
    nearest_tracks = VanillaNearestUsers(1000)
    evaluator = Evaluate(nearest_tracks, 'vanilla_nearest_tracks', K=250, drop=True, random_samples=0)
    evaluator.evaluate(1)


