class Song(object):
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()
        self.song_id = ''
        self.artist_id = ''
        self.publish_time = 0
        self.song_init_plays = 0
        self.language = 0
        self.gender = 0


    def save(self):
        sql = 'insert into song(song_id, artist_id, publish_time, song_init_plays, `language`, gender) VALUES ("%s", "%s", %d, %d, %d, %d)'
        sql_exe = sql % (self.song_id, self.artist_id, int(self.publish_time), int(self.song_init_plays), int(self.language), int(self.gender))
        self.cur.execute(sql_exe)

class Action(object):
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()
        self.user_id = ''
        self.song_id = ''
        self.gmt_create = 0
        self.action_type = 0
        self.ds = 0

    def save(self):
        sql = 'insert into user_action (user_id, song_id, gmt_create, action_type, ds) VALUES ("%s", "%s", %d, %d, %d)'
        sql_exe = sql % (self.user_id, self.song_id, int(self.gmt_create), int(self.action_type), int(self.ds))
        self.cur.execute(sql_exe)

