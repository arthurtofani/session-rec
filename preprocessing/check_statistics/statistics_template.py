import pandas as pd
import numpy as np
from datetime import datetime, timezone

# EC
# diginetica:
    # - train-item-views (some entries (NA))
    # - train-purchases (some entries (NA))

# retailrocket - events

# tmall [Private]
    # - dataset15
    # - lso_train

# Xing 2016 - interactions
PATH = '../../data/xing/xing2016/'
FILE = 'interactions'
# keys
USER_KEY='user_id'
ITEM_KEY='item_id'
TIME_KEY='created_at'
SESSION_KEY='session_id'
TYPE_KEY='interaction_type'
# filters
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5
MIN_USER_SESSIONS = 3
MAX_USER_SESSIONS = 200

# Xing 2017 - interactions

# zalando - lso_train [private]


# Music
# 8tracks - Lso_train / lso_test [private]
# 30music - 30music-200ks
# aotm: [Playlists were randomly distributed to a time span of one year]
    # - playlists-aotm
    # - playlists-aotm_asyear
# lastfm [don’t have raw data]
# nowplaying - nowplaying


def prepare_time(data, time_key=TIME_KEY):
    """Assigns session ids to the events in data without grouping keys"""
    data[time_key] = data[time_key].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    data['tmp'] = 1
    data['tmp'] = data.groupby(SESSION_KEY).tmp.cumsum()
    data[time_key] = data[time_key] + data['tmp']
    del data['tmp']
    return data

def make_sessions(data, session_th=30 * 60, is_ordered=False, user_key=USER_KEY, item_key=ITEM_KEY, time_key=TIME_KEY, session_key=SESSION_KEY):
    """Assigns session ids to the events in data without grouping keys"""
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data[user_key].values[1:] != data[user_key].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data[session_key] = session_ids
    return data



if __name__ == '__main__':

    # updater.dispatcher.add_handler( CommandHandler('status', status) )
    data = pd.read_csv(PATH + FILE + '.csv', sep='\t') #TODO: appropriate seprator
    # remove interactions of type 'delete'
    data = data[data[TYPE_KEY] != 4].copy()
    # remove rows with NA userId
    data = data[~np.isnan(data[USER_KEY])].copy()
    # TODO: appropriate preprocessing for data[TIME_KEY]

    # prepare time format
    data = prepare_time(data, time_key=TIME_KEY)

    # partition interactions into sessions with 30-minutes idle time
    data = make_sessions(data, session_th=30 * 60, is_ordered=False, user_key=USER_KEY, item_key=ITEM_KEY,
                         time_key=TIME_KEY)

    # to delete sessions which the session_id is the same for different users!
    data = data[data[SESSION_KEY].isin(data.groupby(SESSION_KEY)[USER_KEY].nunique()[
                                           (data.groupby(SESSION_KEY)[USER_KEY].nunique() > 1) == False].index)]

    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Original data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('--------------------- Original---')
    print('Num of users: {}'.format(data[USER_KEY].nunique()))
    print('Max num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().max()))
    print('Min num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().min()))
    print('Median num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().median()))
    print('Mean num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().mean()))
    print('Std num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().std()))
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    print('Max num of users\' sessions: {}'.format(sess_per_user.max()))
    print('Min num of users\' sessions: {}'.format(sess_per_user.min()))
    print('Median num of users\' sessions: {}'.format(sess_per_user.median()))
    print('Mean num of users\' sessions: {}'.format(sess_per_user.mean()))
    print('Std num of users\' sessions: {}'.format(sess_per_user.std()))
    print('---------------------')
    # print('Num of sessions: {}'.format(np.count_nonzero(data.groupby(USER_KEY)[SESSION_KEY].nunique())))
    print('Max sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().max()))
    print('Min sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().min()))
    print('Median sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().median()))
    print('Mean sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().mean()))
    print('Std sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().std()))
    print('---------------------')
    print('Num of items: {}'.format(data[ITEM_KEY].nunique()))
    print('Max num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().max()))
    print('Min num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().min()))
    print('Median num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().median()))
    print('Mean num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().mean()))
    print('Std num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().std()))
    # print('Max num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().max()))
    # print('Min num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().min()))
    # print('Median num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().median()))
    # print('Mean num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().mean()))
    # print('Std num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().std()))
    print('---------------------')

    # drop duplicate interactions within the same session
    data.drop_duplicates(subset=[ITEM_KEY, SESSION_KEY, TYPE_KEY], keep='first', inplace=True)

    condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
        [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
        [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT
    count = 1
    while not condition:
        print(count)
        # keep items with >=20 interactions
        item_pop = data[ITEM_KEY].value_counts()
        good_items = item_pop[item_pop >= MIN_ITEM_SUPPORT].index
        data = data[data[ITEM_KEY].isin(good_items)]
        # remove sessions with length < 3
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length >= MIN_SESSION_LENGTH].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # let's keep only returning users (with >= 5 sessions) and remove overly active ones (>=200 sessions)
        sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
        good_users = sess_per_user[(sess_per_user >= MIN_USER_SESSIONS) & (sess_per_user < MAX_USER_SESSIONS)].index
        data = data[data[USER_KEY].isin(good_users)]
        condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
            [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
            [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT
        count += 1

    # output
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('--------------------- Filtered---')
    print('Num of users: {}'.format(data[USER_KEY].nunique()))
    print('Max num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().max()))
    print('Min num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().min()))
    print('Median num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().median()))
    print('Mean num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().mean()))
    print('Std num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().std()))
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    print('Max num of users\' sessions: {}'.format(sess_per_user.max()))
    print('Min num of users\' sessions: {}'.format(sess_per_user.min()))
    print('Median num of users\' sessions: {}'.format(sess_per_user.median()))
    print('Mean num of users\' sessions: {}'.format(sess_per_user.mean()))
    print('Std num of users\' sessions: {}'.format(sess_per_user.std()))
    print('---------------------')
    print('Num of sessions per user: {}'.format(np.count_nonzero(data.groupby(USER_KEY)[SESSION_KEY].nunique())))
    print('Max sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().max()))
    print('Min sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().min()))
    print('Median sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().median()))
    print('Mean sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().mean()))
    print('Std sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().std()))
    print('---------------------')
    print('Num of items: {}'.format(data[ITEM_KEY].nunique()))
    print('Max num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().max()))
    print('Min num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().min()))
    print('Median num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().median()))
    print('Mean num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().mean()))
    print('Std num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().std()))
    # print('Max num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().max()))
    # print('Min num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().min()))
    # print('Median num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().median()))
    # print('Mean num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().mean()))
    # print('Std num of interactions done with an item2: {}'.format(data[ITEM_KEY].value_counts().std()))
    print('---------------------')
