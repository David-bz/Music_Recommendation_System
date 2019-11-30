import requests
import json
import csv
from dataset.dataset_utils import *
import pandas as pd



def lastfm_get(payload):
    API_KEY = '565547727218540858d3a20274c021d2'
    headers = {'user-agent': 'Dataquest'}
    url = 'http://ws.audioscrobbler.com/2.0/'
    payload['api_key'] = API_KEY
    payload['format'] = 'json'
    response = requests.get(url, headers=headers, params=payload)
    return response, response.status_code != 200

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
            if failed: continue
            user_info = req.json()['user']
            if not valid_user(user_info, conditions, metadata_mapping): continue
            if user_info['name'] in metadata_mapping:
                user_info['gender'] = metadata_mapping[user_info['name']]['gender']
                user_info['age'] = metadata_mapping[user_info['name']]['age']
            req, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': username, 'limit' : 1})
            if failed: continue
            loved_tracks_n = int(req.json()['lovedtracks']['@attr']['total'])
            if loved_tracks_n < conditions['loved_tracks']: continue
            req, failed = lastfm_get({'method': 'user.getRecentTracks', 'user': username, 'limit' : 1})
            if failed: continue
            last_activity = int(req.json()['recenttracks']['track'][0]['date']['uts'])
            if last_activity < conditions['last_activity_unix_time']: continue
            req, failed = lastfm_get({'method': 'user.getTopArtists', 'user': username, 'limit' : 1, 'period' : 'overall'})
            if failed: continue
            top_artists_n = int(req.json()['topartists']['@attr']['total'])

            entry = {'id' : id,
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
            if id % 1000 == 0:
                pd.DataFrame.to_csv(users, './dataset/entities/tmp_users.csv.zip', index=False, header=True,
                                    compression='zip')

        except Exception as e:
            print("Username '{}' failed with Error '{}'".format(username, e))

    pd.DataFrame.to_csv(users, './dataset/entities/users.csv.zip', index=False, header=True, compression='zip')


if __name__ == '__main__':
    last_activity_condition = 1420070400 # Unix time (uts) of 1/1/2015, 12:00 AM
    conditions = {'age' : {'min' : 8, 'max' : 90}, 'gender' : ['m', 'f', 'n'], 'playcount': 100, 'loved_tracks' : 40,
                  'last_activity_unix_time' : last_activity_condition}
    with open('./dataset/users_metadata_mapping.json') as f:
        metadata_mapping = json.load(f)
    with open('./dataset/lastfm_45167_usernames.txt') as f:
        content = f.read()
    exec("usernames = " + content)
    with open('./dataset/lastfm_5261_israeli_usenames.txt') as f:
        content = f.read()
    exec("israelis = " + content)
    usernames = usernames.union(israelis)
    generate_users(usernames, conditions, metadata_mapping)
