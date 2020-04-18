import pandas as pd
import json
import requests
import pylast
import numpy as np
import time
import threading
import logging

def lastfm_get(payload):
    API_KEY = '565547727218540858d3a20274c021d2'
    headers = {'user-agent': 'Dataquest'}
    url = 'http://ws.audioscrobbler.com/2.0/'
    payload['api_key'] = API_KEY
    payload['format'] = 'json'
    response = requests.get(url, headers=headers, params=payload)
    if response.status_code != 200:
        print(response.json())
        # retry..
        response = requests.get(url, headers=headers, params=payload)
        if response.status_code != 200:
            return response, True
    return response, False

def init_lastfm():
    API_KEY = '565547727218540858d3a20274c021d2'
    API_SECRET = '1475ea51a769082e151f885e8d3011e7'
    username = 'orshemesh'
    password_hash = pylast.md5('DavidOr123!')
    return pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET, username=username, password_hash=password_hash)

def israelis_users(usernames, metadata_mapping, net):
    users = pd.read_csv('./entities/users.csv.zip', header=0, compression='zip')

    id, i = len(users), 0
    for username in usernames:
        try:
            print('{} {}'.format(i, username))

            user = net.get_user(username)
            if user == None:
                print('user {} not exist'.format(username))
                continue

            req, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': username, 'limit': 1})
            if failed:
                print('user.getLovedTracks failed for user {}'.format(username))
                continue
            loved_tracks_n = int(req.json()['lovedtracks']['@attr']['total'])

            playcount = user.get_playcount()
            if playcount < 10 and loved_tracks_n < 10:
                print('user {} not active: play count {}, love count {}'.format(username, playcount, loved_tracks_n))
                continue

            # default values for new users
            age = -1
            gender = 'N'
            if username in metadata_mapping.keys():
                age = int(metadata_mapping[username]['age'])
                gender = int(metadata_mapping[username]['gender'])

            req, failed = lastfm_get({'method': 'user.getTopArtists', 'user': username, 'limit' : 1, 'period' : 'overall'})
            if failed:
                print('user.getTopArtists failed for user {}'.format(username))
                continue
            top_artists_n = int(req.json()['topartists']['@attr']['total'])

            req, failed = lastfm_get({'method': 'user.getRecentTracks', 'user': username, 'limit': 1})
            if failed:
                continue
            last_activity = int(req.json()['recenttracks']['track'][0]['date']['uts'])

            entry = {
                'id': id,
                'lastfm_username': user.get_name(),
                'age': age,
                'country': user.get_country().get_name(),
                'gender': gender,
                'playcount': user.get_playcount(),
                'loved_tracks': loved_tracks_n,
                'artists_count': top_artists_n,
                'registration_unix_time': user.get_unixtime_registered(),
                'last_activity_unix_time': last_activity,
                'subscriber': user.get_registered()
            }
            users = users.append(entry, ignore_index=True)
            id += 1
            i +=1
            if id % 500 == 0:
                pd.DataFrame.to_csv(users, './entities/tmp_users.csv.zip', index=False, header=True,
                                    compression='zip')
                with open('./entities/remained_{}_usernames.txt'.format(len(usernames)), 'w+') as f:
                    f.write(str(usernames))
        except Exception as e:
            print("Username '{}' failed with Error '{}'".format(username, e))
    pd.DataFrame.to_csv(users, './entities/users.csv.zip', index=False, header=True, compression='zip')

class Songs():
    def __init__(self):
        self.songs = pd.DataFrame(columns=['id', 'name', 'artist'])
        self.lock = threading.Lock()

    def get_song_id(self,song_name,artist):
        song_id = -1
        self.lock.acquire()
        try:
            try:
                song_id = self.songs[(self.songs.name == song_name) & (self.songs.artist == artist)].id.values[0]
            except:
                song_id = len(self.songs)
                self.songs = self.songs.append({'id': song_id, 'name': song_name, 'artist': artist}, ignore_index=True)
                if len(self.songs) % 10000 == 0:
                    self.dump_data(True)
        except Exception as e:
            logging.error('Exception get_song_id({},{}): {}'.format(song_name,artist,e))
        self.lock.release()
        return song_id

    def songs_count(self):
        self.lock.acquire()
        count = len(self.songs)
        self.lock.release()
        return count

    def dump_data(self, temp=False):
        if temp:
            pd.DataFrame.to_csv(self.songs, './entities/tmp_songs.csv.zip', index=False, header=True, compression='zip')
        else:
            pd.DataFrame.to_csv(self.songs, './entities/songs.csv.zip', index=False, header=True, compression='zip')

class Love():
    def __init__(self):
        self.love = pd.DataFrame(columns=['user_id', 'song_id', 'timestamp'])
        self.lock = threading.Lock()

    def addList(self, user_id, songs : Songs, songs_list):
        self.lock.acquire()
        try:
            for played_song in songs_list:
                song = played_song[0]
                song_name = song.get_name()
                song_artist = song.get_artist()
                timestamp = played_song[2]
                self._add(user_id, songs.get_song_id(song_name,song_artist), timestamp)
        except Exception as e:
            logging.error('Exception addList({},_): {}'.format(user_id,e))
        self.lock.release()

    def _add(self, user_id, song_id, timestamp):
        try:
            self.love = self.love.append({'user_id': user_id, 'song_id': song_id, 'timestamp': timestamp}, ignore_index=True)
            if len(self.love) % 10000 == 0:
                self.dump_data(True)
        except Exception as e:
            logging.error('Exception add({},{},{}): {}'.format(user_id, song_id, timestamp, e))

    def dump_data(self, temp=False):
        if temp:
            pd.DataFrame.to_csv(self.love, './relations/tmp_love.csv.zip', index=False, header=True, compression='zip')
        else:
            pd.DataFrame.to_csv(self.love, './relations/love.csv.zip', index=False, header=True, compression='zip')

def get_users_songs_thread(net : pylast.LastFMNetwork, users, songs: Songs, love: Love, ids, thread_num):
    id = -1
    try:
        for i,id in enumerate(ids):
            username = users[users.id == id].lastfm_username.values[0]
            songs_num = users[users.id == id].loved_tracks.values[0]

            for page in [1,2]:
                if ((page-1)*1000 > songs_num):
                    break
                user = net.get_user(username)
                songs_list = None
                for try_i in [1,2,3,4,5]:
                    try:
                        songs_list = user.get_loved_tracks(limit=1000, page=page)
                        break
                    except Exception as e:
                        logging.info('thread {} try {} id {} page {}: {}'.format(thread_num, try_i, id, page, e))
                        time.sleep(4)
                        continue
                if songs_list is None:
                    logging.error('Failed to get loved songs: thread {}, id {}, page {}'.format(thread_num, id, page))
                    break
                love.addList(id, songs, songs_list)
            if i % 10 == 0:
                logging.info('thread {} id {} out of {}'.format(thread_num, i, len(ids)))
    except Exception as e:
        logging.error('Exception get_users_songs_thread id {}: {}'.format(id,e))


def get_users_songs(threads_num):
    users = pd.read_csv('./entities/israelis_users.csv.zip', header=0, compression='zip')
    songs = Songs()
    love = Love()
    net = init_lastfm()
    start = time.time()

    user_ids_list = np.array_split(users.id.values, threads_num)
    for thread_i in range(threads_num):
        thread = threading.Thread(target=get_users_songs_thread, args=(net, users, songs, love, user_ids_list[thread_i], thread_i))
        thread.start()

    while (threading.active_count() > 1):
        current = time.time()
        logging.info('threads {} songs {} love {} time {}'.format(threading.active_count(), len(songs.songs), len(love.love), current-start))
        time.sleep(10)
    songs.dump_data()
    love.dump_data()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('./entities/log.txt'),
        logging.StreamHandler()
    ]
)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('./entities/log_error.txt'),
        logging.StreamHandler()
    ]
)

get_users_songs(5)
#
# # class Songsfff():
# #     def __init__(self):
# #         self.songs_lock = threading.Lock()
# #         self.songs = pd.DataFrame(columns=['id', 'name', 'artist'])
# #         self.user_loved_song_lock = threading.Lock()
# #         self.user_loved_song = pd.DataFrame(columns=['user_id', 'song_id', 'timestamp'])
# #
# #     def get_song_id(self,song_name,artist):
# #         self.songs_lock.acquire()
# #         try:
# #             try:
# #                 song_id = self.songs[(self.songs.name == song_name) & (self.songs.artist == artist)].id.values[0]
# #             except:
# #                 song_id = len(self.songs)
# #             self.songs = self.songs.append({'id': song_id, 'name': song_name, 'artist': artist}, ignore_index=True)
# #             if len(self.songs) % 1000 == 0:
# #                 pd.DataFrame.to_csv(self.songs, './entities/tmp_songs.csv.zip', index=False, header=True,
# #                                     compression='zip')
# #         except Exception as e:
# #             print('Exception get_song_id({},{}): {}'.format(song_name,artist,e))
# #         self.songs_lock.release()
# #         return song_id
# #
# #     def songs_num(self):
# #         self.songs_lock.acquire()
# #         num = len(self.songs)
# #         self.songs_lock.release()
# #         return num
# #
# #     def user_loved_song_num(self):
# #         self.user_loved_song_lock.acquire()
# #         num = len(self.user_loved_song)
# #         self.user_loved_song_lock.release()
# #         return num
# #
# #     def add_song_to_user(self,user_id,song_name,artist,timestamp):
# #         songs_id = self.get_song_id(song_name, artist)
# #
# #         self.user_loved_song_lock.acquire()
# #         try:
# #             self.user_loved_song = self.user_loved_song.append({'user_id': user_id,
# #                                                                 'song_id': songs_id,
# #                                                                 'timestamp': timestamp}, ignore_index=True)
# #             if len(self.user_loved_song) % 1000 == 0:
# #                 pd.DataFrame.to_csv(self.user_loved_song, './entities/tmp_user_loved_song.csv.zip', index=False, header=True,
# #                                     compression='zip')
# #         except Exception as e:
# #             print('Exception add_song_to_user({},{},{},{}): {}'.format(user_id,song_name,artist,timestamp, e))
# #         self.user_loved_song_lock.release()
# #
# #     def get_songs(self):
# #         self.lock.acquire()
# #         res = self.songs
# #         self.lock.release()
# #         return res
#
#
# # def get_user_songs(songs : Songs, username):
# #     user = net.get_user(username)
# #     for page in [1,2]:
# #         songs_list = None
# #         for try_i in range(4):
# #             try:
# #                 songs_list = user.get_loved_tracks(limit=1000, page=page)
# #                 break
# #             except:
# #                 time.sleep(10)
# #                 continue
# #         if songs_list is None:
# #             continue
# #         for i,played_song in enumerate(songs_list):
# #             song = played_song[0]
# #             song_name = song.get_name()
# #             song_artist = song.get_artist()
# #             song_timestamp = played_song[2]
# #             songs.add_song_to_user(username,song_name,song_artist,song_timestamp)
#
# # start = time.time()
# # songs = Songs()
# # threading.Thread(target=get_user_songs, args=(songs, 'PaulieFowl')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'dollarandcents')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'ElRet')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'LiekOmgItsMusic')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Serpent_axed')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'TheCyb')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Hushasha40')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'viathin12')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Dr_Ernst')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'eighto')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'supajuj')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Vinyltechnician')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Etuz')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'kolyuchko')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'allvoxman')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'InKursion')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Viugtor')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'WinoForce')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'satyros2')).start()
# # threading.Thread(target=get_user_songs, args=(songs, 'Nishikido_Nikka')).start()
#
# # while (threading.active_count() > 1):
# #     print('threads num {} songs num {} user_loved_song_num {}'.format(threading.active_count(), songs.songs_num(), songs.user_loved_song_num()))
# #     time.sleep(3)
# #
# # end = time.time()
# # print('end in {} songs num {} user_loved_song_num {}'.format(end - start, songs.songs_num(), songs.user_loved_song_num()))
# #
# #
# def get_song_info(songs,page,songs_list,start,end):
#     for i,played_song in enumerate(songs_list[start:end]):
#         song = played_song[0]
#         try:
#             info = song.get_info()
#             songs.add_song(page, i + start, info)
#         except Exception as e:
#             print('error in page {} song {} error {}'.format(page,start+i,e))
#
# def time_songs_loved(songs,page):
#     print('loved: page {} starts'.format(page))
#     start = time.time()
#     songs_list = user.get_loved_tracks(limit=1000, page=page)
#
#     x1 = threading.Thread(target=get_song_info, args=(songs, page, songs_list, 0,200)).start()
#     x2 = threading.Thread(target=get_song_info, args=(songs, page, songs_list, 200,400)).start()
#     x3 = threading.Thread(target=get_song_info, args=(songs, page, songs_list, 400,600)).start()
#     x4 = threading.Thread(target=get_song_info, args=(songs, page, songs_list, 600,800)).start()
#     x5 = threading.Thread(target=get_song_info, args=(songs, page, songs_list, 800,1000)).start()
#
#     end = time.time()
#     print('loved: page {} time {} len {}'.format(page, end - start, len(songs_list)))
#
# def time_songs_recent(page):
#     print('recent: page {} starts'.format(page))
#     start = time.time()
#     songs = user.get_recent_tracks(limit=999, page=page)
#     # songs = user.get_loved_tracks(limit=1000, page=page)
#     end = time.time()
#     print('recent: page {} time {} len {}'.format(page, end - start, len(songs)))
#
# def time_songs_loved_old(page):
#     start = time.time()
#     for i in range(1,21):
#         res, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': 'AAAzz', 'page': (page-1)*20 + i})
#         if failed:
#             print('retry..')
#             res, failed = lastfm_get({'method': 'user.getLovedTracks', 'user': 'AAAzz', 'page': (page-1)*20 + i})
#             if failed:
#                 print('user.getLovedTracks failed')
#     end = time.time()
#     print('old loved: page {} time {} len {}'.format(page, end - start, 1000))
#
# def time_songs_recent_old(page):
#     start = time.time()
#     for i in range(1,21):
#         res, failed = lastfm_get({'method': 'user.getRecentTracks', 'user': 'AAAzz', 'page': (page-1)*20 + i})
#         if failed:
#             print('retry..')
#             res, failed = lastfm_get({'method': 'user.getRecentTracks', 'user': 'AAAzz', 'page': (page-1)*20 + i})
#             if failed:
#                 print('user.getLovedTracks failed')
#     end = time.time()
#     print('old recent: page {} time {} len {}'.format(page, end - start, 1000))
#
# # songs = Songs()
# # x1 = threading.Thread(target=time_songs_loved, args=(songs, 1))
# # x2 = threading.Thread(target=time_songs_loved, args=(songs, 2))
# # x3 = threading.Thread(target=time_songs_loved, args=(songs, 3))
# # x4 = threading.Thread(target=time_songs_loved, args=(songs, 4))
# # start1 = time.time()
# # x1.start()
# # x2.start()
# # x3.start()
# # x4.start()
# # print('All started')
# # while (threading.active_count() > 1):
# #     print('actives thread {}, num_songs {}'.format(threading.active_count(), songs.songs_num()))
# #     time.sleep(2)
# # end1 = time.time()
#
# # print('all done in {} num_songs {}'.format(end1 - start1, songs.songs_num()))
# # print(songs.get_songs())
# # time_songs_loved(1)
# # time_songs_loved(2)
#
# # time_songs_recent(1)
# # time_songs_recent(2)
#
# # time_songs_loved_old(1)
# # time_songs_loved_old(2)
#
# # time_songs_recent_old(1)
# # time_songs_recent_old(2)
# #
# # songs = user.get_recent_tracks(limit=10)
# # played_song = songs[0]
# # song = played_song[0]
# # song.get_info()
# # timestamp = played_song[3]
# # playcount = song.get_playcount()
# # print(playcount)
# # #
# #
# # songs = user.get_loved_tracks(limit=10, page=1)
# #
# # start = time.time()
# # for played_song in songs:
# #     song = played_song[0]
# #     print(song.get_info())
# # end = time.time()
# # print(end-start)
# #
# # songs = user.get_recent_tracks(limit=10, page=1)
# #
# # start = time.time()
# # for played_song in songs:
# #     song = played_song[0]
# #     print(song.get_info())
# # end = time.time()
# # print(end-start)
#
# # with open('./users_metadata_mapping.json') as f:
# #     metadata_mapping = json.load(f)
# # israelis = set()
# # with open('./lastfm_5261_israeli_usenames.txt') as f:
# #     content = f.read()
# # exec("israelis = " + content)
# # israelis_users(israelis, metadata_mapping, net)
