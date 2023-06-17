from .. import db

class Tweet(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    datetime = db.Column(db.String(200))
    tweet_id = db.Column(db.String(200))
    username = db.Column(db.String(200))
    text = db.Column(db.String(2000))

class Preprocess(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    datetime = db.Column(db.String(200))
    tweet_id = db.Column(db.String(200))
    username = db.Column(db.String(200))
    text = db.Column(db.String(2000))
    remove_user = db.Column(db.String(2000))
    text_cleaning = db.Column(db.String(2000))
    case_folding = db.Column(db.String(2000))
    tokenizing = db.Column(db.String(2000))
    stop_words = db.Column(db.String(2000))
    stemming = db.Column(db.String(2000))

class Labeling(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    datetime = db.Column(db.String(200))
    tweet_id = db.Column(db.String(200))
    username = db.Column(db.String(200))
    text = db.Column(db.String(2000))
    remove_user = db.Column(db.String(2000))
    text_cleaning = db.Column(db.String(2000))
    case_folding = db.Column(db.String(2000))
    tokenizing = db.Column(db.String(2000))
    stop_words = db.Column(db.String(2000))
    stemming = db.Column(db.String(2000))
    score = db.Column(db.String(2000))
    compound = db.Column(db.Integer)
    sentimen = db.Column(db.String(2000))
    