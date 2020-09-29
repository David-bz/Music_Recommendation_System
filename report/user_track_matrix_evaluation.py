from dataset.user_item_matrix import *
from models.evaluate import *


def metrics_function_of_w2():
    user_track_matrix = userTrackMatrix(verbose=True, drop=False)
    np.random.seed(1)
    users = list(np.random.choice(user_track_matrix.shape[0], 250))

    # create score functions - for each implicit / explicit relation increase the score by 1 or 5 respectively
    w1 = 1
    implicit_promotion_step = lambda obj, i, j: obj.mat[i, j] + w1
    user_track_matrix.process_played_events(implicit_promotion_step)
    implicit_only_mat = user_track_matrix.mat

    x = [w2 for w2 in range(0, 26)]
    loved_map, played_map, related_map = [], [], []
    loved_pre, played_pre, related_pre = [], [], []
    loved_rec, played_rec, related_rec = [], [], []
    for w2 in x:
        print('calculate w1 {} w2 {}'.format(w1, w2))
        user_track_matrix.mat = implicit_only_mat
        explicit_promotion_step = lambda obj, i, j: obj.mat[i, j] + w2
        user_track_matrix.process_loved_events(explicit_promotion_step)
        user_track_matrix.matrix_factorize()

        vanilla_mf = Vanilla_MF(user_track_matrix.MF_users, user_track_matrix.MF_tracks)
        eval = Evaluate(vanilla_mf, 'vanilla_mf_w2_{}'.format(w2), K=250, drop=False, random_samples=0, verbose=True)
        res_dict = eval.evaluate(users)
        res_dict = res_dict['average']

        loved_map.append(res_dict['MAP']['loved_score'])
        played_map.append(res_dict['MAP']['played_score'])
        related_map.append(res_dict['MAP']['related_score'])

        loved_pre.append(res_dict['P@K']['loved_score'])
        played_pre.append(res_dict['P@K']['played_score'])
        related_pre.append(res_dict['P@K']['related_score'])

        loved_rec.append(res_dict['R@K']['loved_score'])
        played_rec.append(res_dict['R@K']['played_score'])
        related_rec.append(res_dict['R@K']['related_score'])

        print('finished w1 {} w2 {}\n'.format(loved_map[-1], w1, w2))

    zipped = zip(x,
                 loved_map, played_map, related_map,
                 loved_pre, played_pre, related_pre,
                 loved_rec, played_rec, related_rec)
    df = pd.DataFrame(list(zipped), columns=['w2',
                                             'loved MAP', 'played MAP', 'related MAP',
                                             'loved P@K', 'played P@K', 'related P@K',
                                             'loved R@K', 'played R@K', 'related R@K'])
    pd.DataFrame.to_csv(df, generate_path('report/user_track_matrix/metrics_as_function_of_w2.csv'),
                        index=False, header=True)


def metrics_as_function_of_n_components():
    user_track_matrix = userTrackMatrix(verbose=True, drop=False)
    np.random.seed(1)
    users = list(np.random.choice(user_track_matrix.shape[0], 250))
    user_track_matrix.load()

    x = [n_components for n_components in range(1, 21)]
    loved_map, played_map, related_map = [], [], []
    loved_pre, played_pre, related_pre = [], [], []
    loved_rec, played_rec, related_rec = [], [], []
    users_var, tracks_var = [], []
    for n_components in x:
        print('calculate {} components'.format(n_components))
        user_track_matrix.matrix_factorize(n_components=n_components)

        vanilla_mf = Vanilla_MF(user_track_matrix.MF_users, user_track_matrix.MF_tracks)
        eval = Evaluate(vanilla_mf, 'vanilla_mf_n_components'.format(n_components), K=250, drop=False,
                        random_samples=0, verbose=True)
        res_dict = eval.evaluate(users)
        res_dict = res_dict['average']

        loved_map.append(res_dict['MAP']['loved_score'])
        played_map.append(res_dict['MAP']['played_score'])
        related_map.append(res_dict['MAP']['related_score'])

        loved_pre.append(res_dict['P@K']['loved_score'])
        played_pre.append(res_dict['P@K']['played_score'])
        related_pre.append(res_dict['P@K']['related_score'])

        loved_rec.append(res_dict['R@K']['loved_score'])
        played_rec.append(res_dict['R@K']['played_score'])
        related_rec.append(res_dict['R@K']['related_score'])

        users_var.append(user_track_matrix.MF_users.var())
        tracks_var.append(user_track_matrix.MF_tracks.var())

        print('finished components {}\n'.format(n_components))

    zipped = zip(x,
                 loved_map, played_map, related_map,
                 loved_pre, played_pre, related_pre,
                 loved_rec, played_rec, related_rec,
                 users_var, tracks_var)
    df = pd.DataFrame(list(zipped), columns=['n_components',
                                             'loved MAP', 'played MAP', 'related MAP',
                                             'loved P@K', 'played P@K', 'related P@K',
                                             'loved R@K', 'played R@K', 'related R@K',
                                             'users var', 'tracks var'])
    pd.DataFrame.to_csv(df, generate_path('report/user_track_matrix/metrics_error_as_func_of_n_components.csv'),
                        index=False, header=True)


if __name__ == '__main__':
    # metrics_function_of_w2()
    metrics_as_function_of_n_components()
