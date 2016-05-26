import pymysql
import models

conn = pymysql.connect(host='localhost', port=3306, user='user', password='password', db='music')



#
# with open('mars_tianchi_songs.csv','r') as songs:
#     songs = songs.readlines()
#     i = 0.
#     l = len(songs)
#     for song in songs:
#         song = song[:-1]
#         song_set = song.split(',')
#         if len(song_set) == 6:
#             s = models.Song(conn)
#             s.song_id = song_set[0]
#             s.artist_id = song_set[1]
#             s.publish_time = int(song_set[2])
#             s.song_init_plays = int(song_set[3])
#             s.language = int(song_set[4])
#             s.gender = int(song_set[5])
#             s.save()
#         i += 1
#         print(str(i/l))

with open('mars_tianchi_user_actions.csv', 'r') as actions:
    actions = actions.readlines()
    i = 0.
    l = len(actions)
    for action in actions:
        action = action[:-1]
        action_set = action.split(',')
        if len(action_set) == 5:
            a = models.Action(conn)
            a.user_id = action_set[0]
            a.song_id = action_set[1]
            a.gmt_create = action_set[2]
            a.action_type = action_set[3]
            a.ds = action_set[4]
            a.save()
        i += 1
        print(str(i/l))

conn.commit()
