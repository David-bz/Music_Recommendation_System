from dataset.data_scaling import *
import scipy.sparse as sps
from sklearn.decomposition import *
from utils import generate_path
import json


class userTrackMatrix:
    def __init__(self, verbose=True, drop=False):
        self.verbose = verbose
        self.data = Dataset()
        self.users = self.data.users
        self.tracks = self.data.tracks
        self.shape = (len(self.users), len(self.tracks))
        self.mat = sps.lil_matrix(self.shape)
        self.drop = drop
        self.drop_dict = None
        self.relation_dir = 'dataset/relations/'

        self.loved = pd.read_csv(generate_path('/dataset/relations/loved.csv.zip'), header=0, compression='zip')
        self.played = pd.read_csv(generate_path('/dataset/relations/recently_played.csv.zip'), header=0, compression='zip')

        self.n_components = 0  # symbolise that both self.MF_users & self.MF_tracks didn't set yet
        if self.verbose:
            print('create users-tracks matrix {}'.format(self.shape))
            print('{} played events and {} loved events to process'.format(len(self.played), len(self.loved)))

    def get_item_count(self, table, attribute, val):
        return len(table[table[attribute] == val])

    def get_user_play_count(self, user_id):
        return self.get_item_count(self.played, 'user_id', user_id)

    def get_user_love_count(self, user_id):
        return self.get_item_count(self.loved, 'user_id', user_id)

    def get_track_play_count(self, track_id):
        return self.get_item_count(self.played, 'track_id', track_id)

    def get_track_love_count(self, track_id):
        return self.get_item_count(self.loved, 'track_id', track_id)

    def process_table_events(self, table, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        bucket_size = 0
        if self.verbose:
            bucket_size = len(table) // 100

        for index, row in table.iterrows():
            if self.drop_dict is not None \
                    and str(row.user_id) in self.drop_dict.keys() \
                    and row.track_id in self.drop_dict[str(row.user_id)]:
                continue

            self.mat[row.user_id, row.track_id] = score_fun(self, row.user_id, row.track_id)
            if self.verbose and index % bucket_size == 0:
                print('done {}%: {} out of {}'.format(index // bucket_size, index, len(table)))

    def fill_drop_dict(self, drop_track_per):
        self.drop_dict, count = dict(), 0
        for user in range(self.shape[0]):
            num_to_drop = self.get_user_love_count(user) // drop_track_per
            if num_to_drop <= 0:
                continue

            self.drop_dict[str(user)] = []
            tracks_to_drop = np.random.choice(self.get_user_loved_tracks(user), num_to_drop)
            for track in tracks_to_drop:
                count += 1
                self.drop_dict[str(user)].append(int(track))

                if self.verbose and count % 100 == 0:
                    print('{} tracks propped'.format(count))

    def process_loved_events(self, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        if self.drop:
            self.fill_drop_dict(20)

        self.process_table_events(self.loved, score_fun)
        if self.verbose:
            print('sparsity {}'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1])))

    def process_played_events(self, score_fun):
        # score_fun: a function (self * user_id * track_id) -> (score)
        self.process_table_events(self.played, score_fun)
        if self.verbose:
            print('sparsity {}'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1])))

    def get_drop_prefix(self):
        if self.drop:
            return 'drop_'
        else:
            return ''

    def get_paths(self):
        npz_path = self.relation_dir + self.get_drop_prefix() + 'user_track_matrix.npz'
        users_path = self.relation_dir + self.get_drop_prefix() + 'mf_users.npy'
        tracks_path = self.relation_dir + self.get_drop_prefix() + 'mf_tracks.npy'
        return npz_path, users_path, tracks_path

    def dump(self):
        npz_path, users_path, tracks_path = self.get_paths()
        coo_matrix = self.mat.tocoo()
        sps.save_npz(generate_path(npz_path), coo_matrix)
        if self.n_components > 0:
            np.save(generate_path(users_path), self.MF_users)
            np.save(generate_path(tracks_path), self.MF_tracks)
        if self.drop_dict is not None:
            with open(generate_path(self.relation_dir + 'drop_dict.json'), 'w') as fp:
                json.dump(self.drop_dict, fp)

    def load_mat(self, npz_path):
        self.mat = sps.load_npz(generate_path(npz_path)).tolil()

    def load(self):
        npz_path, users_path, tracks_path = self.get_paths()
        self.load_mat(npz_path)
        try:
            self.MF_tracks = np.load(generate_path(tracks_path))
            self.MF_users = np.load(generate_path(users_path))
            self.n_components = len(self.MF_users[0])
            if self.drop:
                with open(generate_path(self.relation_dir + 'drop_dict.json')) as json_file:
                    self.drop_dict = json.load(json_file)

            if self.verbose:
                print('sparsity {}, {} components'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1]),
                                                          self.n_components))
        except Exception as e:
            print(e)

    def matrix_factorize(self, n_components=12, init='nndsvd', l1_ratio=0.5, tol=1e-4, alpha=0.):
        self.n_components = n_components
        model = NMF(n_components=n_components, init=init, random_state=0,
                    verbose=self.verbose, max_iter=400, l1_ratio=l1_ratio,
                    tol=tol, alpha=alpha)
        self.MF_users = model.fit_transform(self.mat)
        self.MF_tracks = model.components_
        if self.verbose:
            print('{} components reconstruction error {}'.format(n_components, model.reconstruction_err_))
            print('sparsity {}'.format(self.mat.nnz / (self.mat.shape[0] * self.mat.shape[1])))

    def get_mf_score(self, user, track):
        return sum([self.MF_users[user, i] * self.MF_tracks[i, track] for i in range(self.n_components)])

    def get_user_loved_tracks(self, user):
        # related_tracks contains all the tracks that this user loved
        tracks = list(self.loved[self.loved.user_id == user].track_id.values)
        if self.drop and user in self.drop_dict.keys():
            tracks = [track for track in tracks if track not in self.drop_dict[str(user)]]
        return tracks

    def get_user_played_tracks(self, user):
        # related_tracks contains all the tracks that this user played
        tracks = list(self.played[self.played.user_id == user].track_id.values)
        if self.drop and user in self.drop_dict.keys():
            tracks = [track for track in tracks if track not in self.drop_dict[str(user)]]
        return tracks

    def get_user_related_tracks(self, user):
        # related_tracks contains all the tracks that this user loved or played
        return self.get_user_loved_tracks(user) + self.get_user_played_tracks(user)

    def get_user_dropped_tracks(self, user):
        # when in drop mode, get the list for a certain user
        return self.drop_dict.get(str(user), []) if self.drop else []


    def evaluate_mf(self, samples=200):
        count = 0
        sum = 0
        for user in np.random.choice(self.shape[0], samples):
            # score[i] is the score of track_id == i
            scores = np.dot(self.MF_users[user, :], self.MF_tracks)
            # positions[i] is the i'th worst track by this user
            positions = np.argsort(scores)
            # related_tracks contains all the tracks that this user loved / played
            related_tracks = self.get_user_related_tracks(user)

            # calculate the average positions of the related tracks
            avg_position = 0
            related_tracks_len = len(related_tracks)
            for track in related_tracks:
                avg_position += (np.argwhere(positions == track)[0][0] / related_tracks_len)
            # the score of this user is average positions divide the number of tracks
            # so if avg_position is 500K, divide it by 609K the score is ~0.82
            # so if avg_position is 300K, divide it by 609K the score is ~0.49
            # remember that `positions` is sorted from the worst to the best track
            user_score = avg_position / self.shape[1]

            if self.verbose:
                print('user {}, score {}, avg position {}, {} related tracks'.format(
                    user, user_score, int(avg_position), len(related_tracks)))
            sum += user_score
            count += 1

        print('score {}'.format(sum / count))


if __name__ == '__main__':
    user_track_matrix = userTrackMatrix(drop=True)

    # create a trivial score function - for each event increase the score by 1
    played_promotion_step = lambda obj, i, j: obj.mat[i, j] + 1
    loveded_promotion_step = lambda obj, i, j: obj.mat[i, j] + 5

    user_track_matrix.process_loved_events(loveded_promotion_step)
    user_track_matrix.process_played_events(played_promotion_step)
    user_track_matrix.matrix_factorize()
    user_track_matrix.evaluate_mf()
    user_track_matrix.dump()
