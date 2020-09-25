from dataset.user_item_matrix import *

def map_as_function_of_w2():
    user_track_matrix = userTrackMatrix(verbose=False, drop=False)
    np.random.seed(1)
    users = np.random.choice(user_track_matrix.shape[0], 250)

    # create score functions - for each implicit / explicit relation increase the score by 1 or 5 respectively
    w1 = 1
    implicit_promotion_step = lambda obj, i, j: obj.mat[i, j] + w1
    user_track_matrix.process_played_events(implicit_promotion_step)
    implicit_only_mat = user_track_matrix.mat

    x, y = [w2 for w2 in range(0, 26)], []
    for w2 in x:
        print('calculate w1 {} w2 {}'.format(w1, w2))
        user_track_matrix.mat = implicit_only_mat
        explicit_promotion_step = lambda obj, i, j: obj.mat[i, j] + w2
        user_track_matrix.process_loved_events(explicit_promotion_step)
        user_track_matrix.matrix_factorize()
        y.append(round(user_track_matrix.evaluate_mf(users), 4))
        print('MAP {} w1 {} w2 {}\n'.format(y[-1], w1, w2))

    df = pd.DataFrame(list(zip(x, y)), columns=['w2', 'MAP'])
    pd.DataFrame.to_csv(df, generate_path('report/user_track_matrix/MAP_as_function_of_w2.csv'),
                        index=False, header=True)


def map_and_reconstruction_error_as_function_of_n_components():
    user_track_matrix = userTrackMatrix(verbose=False, drop=False)
    np.random.seed(1)
    users = np.random.choice(user_track_matrix.shape[0], 250)

    # create score functions - for each implicit / explicit relation increase the score by 1 or 5 respectively
    implicit_promotion_step = lambda obj, i, j: obj.mat[i, j] + 1
    explicit_promotion_step = lambda obj, i, j: obj.mat[i, j] + 5

    user_track_matrix.process_loved_events(explicit_promotion_step)
    user_track_matrix.process_played_events(implicit_promotion_step)

    x, MAP, reconstruction_errors = [n_components for n_components in range(1, 21)], [], []
    for n_components in x:
        print('calculate {} components'.format(n_components))
        reconstruction_errors.append(round(user_track_matrix.matrix_factorize(n_components=n_components), 2))
        MAP.append(round(user_track_matrix.evaluate_mf(users), 4))
        print('{} components: {} MAP, {}\n'.format(n_components, MAP[-1], reconstruction_errors[-1]))

    df = pd.DataFrame(list(zip(x, MAP, reconstruction_errors)), columns=['n_components', 'MAP', 'reconstruction error'])
    pd.DataFrame.to_csv(df, generate_path('report/user_track_matrix/map_and_reconstruction_error_as_func_of_n_components.csv'),
                        index=False, header=True)


if __name__ == '__main__':
    # map_as_function_of_w2()
    map_and_reconstruction_error_as_function_of_n_components()
