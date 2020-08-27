import requests
import json
import csv
import pandas as pd
import logging
import scipy.sparse as sps
import numpy as np

def lastfm_get(payload):
    API_KEY = '565547727218540858d3a20274c021d2'
    headers = {'user-agent': 'Dataquest'}
    url = 'http://ws.audioscrobbler.com/2.0/'
    payload['api_key'] = API_KEY
    payload['format'] = 'json'
    response = requests.get(url, headers=headers, params=payload)
    if response.status_code != 200:
        print(response.json())
        return response, True
    return response, False


def jprint(obj):
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


def israelis_users_search(required_users_num):
    '''
    :param required_users_num: The amount of users we'd to accumulate in the search.
    This function performs a BFS search among Last.fm's israeli users. We initialize the search with a small group
    of israeli usernames, derived from '30Music' dataset, and from the common israeli first names.

    '''
    last_fm_israelis_users = {'aconomika', 'a-lester', 'AlonMizrahi', 'Alternative321', 'AsyaThawho', 'aviadhadad', 'Ba5tarD', 'blizok', 'BloodRedWoman', 'Bodokid', 'boffin13', 'boo-zi',
                              'brudrex', 'cheall', 'Chenipeper', 'chi22ko', 'danjashwarts', 'deepdarkthinker', 'demiliv', 'DemonHunterD', 'desertdiver', 'dim2kom', 'elguyer', 'F_Nikiforov',
                              'G28', 'GodDamnTheSwan', 'guy120', 'hagarit', 'halnine', 'HarelSkaat', 'Hayim', 'Hilelko', 'Holy_Murder', 'hronometer', 'IL69POPGEJU', 'IL-91', 'ILionAfrakaI',
                              'ILoveAkasha', 'ILOVEDUBMUSIC', 'ILOVEMUSIC11', 'inmost_light', 'isittaken', 'Itamar95', 'jowiisia', 'lammasabacthani', 'lidermanrony', 'liorms', 'magazanik',
                              'malihant', 'Max_Agarkov', 'Mayan89', 'Mc_Natasha', 'Metal_tne_best', 'mikeyrhcp', 'MikLyarych', 'Moon_Drifter76', 'morth66', 'MrPuroresu', 'MysterMania',
                              'naftali10', 'NeriaNa', 'oblivion80', 'obormot', 'Omegamax', 'PacTa-MoHaX', 'pan_denis', 'Pasha89', 'paskol-s', 'PavelZagalsky', 'pluracell3000', 'poetic-killer',
                              'pokutan', 'PsychedelicAcid', 'razterizer', 'rodionm', 'Ronit123', 'royal99', 'ruggerk', 'shalosh4laavoda', 'shaninosh', 'Shorty_17', 'silviaelen', 'Sindri32',
                              'SouthOfHeaven97', 'strummer121', 'Tamyl', 'TristanBlair', 'turkiz', 'urban_wh0re', 'Vodaka', 'ymarder', 'D--Soul', 'abialystok', 'shigaugi', 'JustRap', 'ILGIZA',
                              'sofia_h', 'kiallan', 'victimex', 'gillush', 'smileytpb', 'oranium77', 'Polymeron'}
    israelis_known_names = {'Or', 'David', 'Dorin', 'Adi','Noa','Nimrod','Asaf','Itamar','Daniel','Hai','Alon','Dana','Adir','Yossi','Ido','Amir','Itay','Ariel','Shahaf','Shahar','Tal','Ofek'}
    open_set = last_fm_israelis_users.union(israelis_known_names)
    close_set, active_users = set(), set()

    while (len(active_users) < required_users_num and len(open_set) > 0):
        print('open {}, close {}, active {}'.format(len(open_set), len(close_set), len(active_users)))
        current_user = open_set.pop()

        r,failed = lastfm_get({'method': 'user.getInfo', 'user': current_user})
        if failed:
            # that means this user name does not exist
            close_set.add(current_user)
            continue
        else:
            has_playlists = r.json()['user']['playlists'] != '0'
            has_enough_playcounts = int(r.json()['user']['playcount']) > 100
            israeli = r.json()['user']['country'] == 'Israel'

        r, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': current_user, 'limit':1})
        if failed:
            close_set.add(current_user)
            continue
        else:
            has_enough_loved_songs = int(r.json()['lovedtracks']['@attr']['total']) > 20

        active_user = has_playlists or has_enough_playcounts or has_enough_loved_songs

        if active_user and israeli:
            active_users.add(current_user)

        r,failed = lastfm_get({'method': 'user.getFriends', 'user': current_user})
        if failed:
            close_set.add(current_user)
            continue

        for friend in r.json()['friends']['user']:
            if friend['country'] == 'Israel':
                if not friend['name'] in close_set:
                    open_set.add(friend['name'])
        close_set.add(current_user)
    with open('./data/Israelis_acite_users_{}.txt'.format(len(active_users)), 'w') as f:
        f.write(str(active_users))
    print(active_users)


def valid_user(user_info, conditions, metadata_mapping={}):
    if user_info['name'] in metadata_mapping:
        age = metadata_mapping[user_info['name']]['age']
        if not conditions['age']['min'] < int(age) < conditions['age']['max']:
            return False
    if user_info['country'] == "" or user_info['country'] == "None":
        return False
    if int(user_info['playcount']) < conditions['playcount']:
        return False

    return True


def generate_users(usernames, conditions, metadata_mapping={}):
    users = pd.DataFrame(columns=['id', 'lastfm_username', 'age', 'country', 'gender', 'playcount', 'loved_tracks',
                                  'artists_count', 'registration_unix_time', 'last_activity_unix_time', 'subscriber'])
    id = 0
    N, count = len(usernames), 0
    while len(usernames) > 0:
        try:
            username = usernames.pop()
            count += 1
            print(count, id)
            req, failed = lastfm_get({'method' : 'user.getInfo', 'user' : username})
            if failed:
                continue
            user_info = req.json()['user']
            if not valid_user(user_info, conditions, metadata_mapping): continue
            if user_info['name'] in metadata_mapping:
                user_info['gender'] = metadata_mapping[user_info['name']]['gender']
                user_info['age'] = metadata_mapping[user_info['name']]['age']
            req, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': username, 'limit' : 1})
            if failed:
                continue
            loved_tracks_n = int(req.json()['lovedtracks']['@attr']['total'])
            if loved_tracks_n < conditions['loved_tracks']: continue
            req, failed = lastfm_get({'method': 'user.getRecentTracks', 'user': username, 'limit' : 1})
            if failed:
                continue
            last_activity = int(req.json()['recenttracks']['track'][0]['date']['uts'])
            if last_activity < conditions['last_activity_unix_time']: continue
            req, failed = lastfm_get({'method': 'user.getTopArtists', 'user': username, 'limit' : 1, 'period' : 'overall'})
            if failed:
                continue
            top_artists_n = int(req.json()['topartists']['@attr']['total'])

            entry = {
                'id' : id,
                'lastfm_username' : user_info['name'],
                'age' : int(user_info['age']),
                'country' : user_info['country'],
                'gender' : user_info['gender'],
                'playcount' : int(user_info['playcount']),
                'loved_tracks' : loved_tracks_n,
                'artists_count' : top_artists_n,
                'registration_unix_time' : int(user_info['registered']['unixtime']),
                'last_activity_unix_time' : last_activity,
                'subscriber' : int(user_info['subscriber'])
            }
            users = users.append(entry, ignore_index=True)
            id += 1
            if id % 500 == 0:
                pd.DataFrame.to_csv(users, './dataset/entities/tmp_users.csv.zip', index=False, header=True,
                                    compression='zip')
                with open('./dataset/entities/remained_{}_usernames.txt'.format(len(usernames)), 'w+') as f:
                    f.write(str(usernames))
        except Exception as e:
            print("Username '{}' failed with Error '{}'".format(username, e))

    pd.DataFrame.to_csv(users, './dataset/entities/users.csv.zip', index=False, header=True, compression='zip')


def save_lil_matrix(matrix, path):
    coo_matrix = matrix.tocoo()
    sps.save_npz(path, coo_matrix)


def add_song(songs, love_matrix, song, user_id, users_num):
    # trying to do a lookup, if song exist return (songs is defined by both name + artist)
    song_entry = songs[ (songs.song_name == song['name']) & (songs.artist_name == song['artist']['name']) ]
    if len(song_entry) != 0:
        print("song `{}` exist".format(song['name']))

        # update love matrix appropriately and return
        song_id = song_entry.id.values[0]
        love_matrix[user_id, song_id] += 1
        return songs

    res, failed = lastfm_get({'method' : 'track.getInfo', 'autocorrect' : 1,
                             'track' : song['name'], 'artist': song['artist']['name']})
    if failed or 'error' in res.json().keys():
        logging.error('track.getInfo failed: song `{}`, artist `{}`, user id {}, error {}'.format(
            song['name'], song['artist']['name'], user_id, res.json()['error']))
        return songs

    track = res.json()['track']
    entry = {
        'id' : len(songs),
        'song_name' : track['name'],
        'artist_name' : track['artist']['name'],
        'listeners' : track['listeners'],
        'playcount' : track['playcount'],
        'duration' : track['duration']
    }
    songs = songs.append(entry, ignore_index=True)

    if songs.id.max() % 1000 == 0:
        # store backups
        save_lil_matrix(love_matrix, './relations/tmp_love.npz')
        pd.DataFrame.to_csv(songs, './entities/tmp_songs.csv.zip', index=False, header=True, compression='zip')
        with open('./songs_until_user.txt', 'w') as f:
            f.write(str(user_id))

        # increase matrix's size
        new_songs_num = songs.id.max() + 1000
        love_matrix.resize((users_num, new_songs_num))

    love_matrix[user_id, entry['id']] += 1

    print("song: id {} name `{}`".format(entry['id'], song['name']))
    return songs

def loved_songs_init_state(user_id_to_start = 0, path_to_love_matrix = '', path_to_tmp_songs = ''):
    if user_id_to_start != 0: assert path_to_love_matrix != '' and path_to_tmp_songs != ''

    users = pd.read_csv('./entities/users.csv.zip', header=0, compression='zip')

    if user_id_to_start == 0:
        songs = pd.DataFrame(columns=['id', 'song_name', 'artist_name', 'listeners', 'playcount', 'duration'])
                                  # 'artist_name', 'artist_listeners', 'artist_playcount',
                                  # 'album_name', 'album_listeners', 'album_playcount', 'album_position'])
        love_matrix = sps.lil_matrix((len(users), 1000)) # we'de gradually increase the songs number
    else:
        songs = pd.read_csv(path_to_tmp_songs, header=0, compression='zip')
        love_matrix = sps.load_npz(path_to_love_matrix).tolil()
        assert love_matrix.shape[0] == len(users)
        assert love_matrix.shape[1] > songs.id.max()

        # clean the loved tracks from this user
        for user_i in range(user_id_to_start, len(users)):
            for song_j in range(songs.id.max() + 1):
                love_matrix[user_i, song_j] = 0

    logging.basicConfig(filename='./entities/log.txt', filemode='w+', format='%(name)s - %(levelname)s - %(message)s')

    return users, songs, love_matrix



def generate_loved_songs(user_id_to_start = 0, path_to_love_matrix = '', path_to_tmp_songs = ''):
    users, songs, love_matrix = loved_songs_init_state(user_id_to_start, path_to_love_matrix, path_to_tmp_songs)

    # iterate over the users
    for user_i in range(user_id_to_start, len(users)):
        try:
            username = users[users.id == user_i].lastfm_username.values[0]
            res, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': username})
            if failed:
                logging.error('user.getLovedTracks failed: username {}'.format(username))
                continue
            print('user {} out of {}'.format(user_i, len(users)))

            # iterate over the pages of songs that the user loved
            loved_songs_pages = int(res.json()['lovedtracks']['@attr']['totalPages'])
            for page_i in range(1, 1+loved_songs_pages):
                if page_i != 1:
                    res, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': username, 'page':page_i})
                    if failed:
                        logging.error('user.getLovedTracks failed: username {}, page {}'.format(username, page_i))
                        continue
                print('user {} page {} out of {}'.format(user_i, page_i, loved_songs_pages))

                # iterate over the songs in the page
                page_songs_list = res.json()['lovedtracks']['track']
                for song_i in range(len(page_songs_list)):
                    songs = add_song(songs, love_matrix, page_songs_list[song_i], user_i, len(users))
        except Exception as e:
            print(e)
            logging.error('Exception raised: {}'.format(e))
            logging.error('Exception: user {}, page {}, song {}'.format(user_i, page_i, song_i))

    save_lil_matrix(love_matrix, './relations/love.npz')
    pd.DataFrame.to_csv(songs, './entities/songs.csv.zip', index=False, header=True, compression='zip')



if __name__ == '__main__':
    # last_activity_condition = 1420070400 # Unix time (uts) of 1/1/2015, 12:00 AM
    # conditions = {'age' : {'min' : 8, 'max' : 90}, 'gender' : ['m', 'f', 'n'], 'playcount': 100, 'loved_tracks' : 40,
    #               'last_activity_unix_time' : last_activity_condition}
    # with open('./dataset/users_metadata_mapping.json') as f:
    #     metadata_mapping = json.load(f)
    # usernames, israelis = set(), set()
    # with open('./dataset/lastfm_45167_usernames.txt') as f:
    #     content = f.read()
    # exec("usernames = " + content)
    # with open('./dataset/lastfm_5261_israeli_usenames.txt') as f:
    #     content = f.read()
    # exec("israelis = " + content)
    # usernames = usernames.union(israelis)
    # generate_users(usernames, conditions, metadata_mapping)

    # generate_loved_songs(76, './relations/tmp_love.npz', './entities/tmp_songs.csv.zip')
    generate_loved_songs()
    pass
