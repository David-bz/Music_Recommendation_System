import requests
import json
import csv


def get_users_from_idomaar():
    with open('./dataset/users.idomaar') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        lastfm_users = set()
        count = 0
        metadata_mapping = {}
        for row in csv_reader:
            count += 1
            user = json.loads(str(row[3]).replace('"age":,', '"age":-1,').replace('"gender":"",', '"gender":"n",').replace('"country":"",', '"country":"XX",'))
            username = user['lastfm_username']
            lastfm_users.add(username)
            metadata_mapping[username] = {'age' : user['age'], 'gender' : user['gender']}
        with open('./dataset/lastfm_{}_usernames.txt'.format(count), 'w+') as f:
            f.write(str(lastfm_users))
        with open('./dataset/users_metadata_mapping.json', 'w+') as json_file:
            json.dump(metadata_mapping, json_file)

