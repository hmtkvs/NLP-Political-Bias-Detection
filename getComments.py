
import tweepy
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream
from datetime import datetime

import mysql.connector
from mysql.connector import Error
from DBHandler import *
import Constants

import json

import emoji

#estudiant credentials
consumer_key = 'JzUtsAgT6ovub38cCxQESpImB'
consumer_secret = 'wmnaQWJwUn6lsAcEcHSWkiNjXnGZdkOa91jiDHIl0TASL9SVK0'
access_token = '1218182611270610950-4aKbUcK9C47QT4pvf98wd4tuYWC0E9'
access_secret = 's9O0h8mZElMkm1yOG9mkwRHWYn6RyHhBbxALJJfUPDX2x'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

auth_api = tweepy.API(auth,  wait_on_rate_limit=True)
print(auth_api)

'''
Collects comments for new tweets.
Returns JSON file and inserts into MySQL
'''

def fetchMysql(n):
    IDs = list()   
    try:
        connection = DBHandler.get_mydb()
        cursor = connection.cursor() 
        print(n)
        for i in range(0,n):
            sql_querry = "select id_tweet, time_tweet FROM tweets order by time_tweet desc LIMIT {}".format(n)
            cursor.execute(sql_querry)
            results = cursor.fetchall()
            IDs.append(results[i][0])
            print("tweet_time: ",results[i][1])
        print("Found ID Length:", len(IDs))
         
    except Error as e:
        print("Error while connecting to MySQL", e)
        
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    return IDs
    
def insertMysql(cm):  
    try:
        print("This is cm", cm[0]['id_tweet'])
        connection = DBHandler.get_mydb()
        cursor = connection.cursor() 
        #!!!!!
        for i in range(0,len(cm)):
            id_user = cm[i]['id_user']
            name_user = cm[i]['name_user']
            
            txt = emoji.demojize(cm[i]['text_comment'])
            txt = txt.replace(":"," ")
            text_comment = ' '.join(txt.split())
            
            time_comment = cm[i]['time_comment']
            id_comment = cm[i]['id_comment']
            id_tweet = cm[i]['id_tweet']

            cursor.execute("INSERT INTO comments(id_user, name_user, text_comment, time_comment, id_comment, id_tweet) VALUES(%s,%s,%s,%s,%s,%s)",(id_user, name_user, text_comment, time_comment, id_comment, id_tweet))
            connection.commit()
    
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def getcomments(tweet_id):
    cm = list()
    for page in tweepy.Cursor(auth_api.search,q='to:cnn', since_id=tweet_id, result_type='recent',timeout=999999,tweet_mode='extended').pages():
       for tweet in page:
            if hasattr(tweet, 'in_reply_to_status_id_str'):
                if (str(tweet.in_reply_to_status_id_str) == str(tweet_id)):
                    time = tweet.created_at.strftime("%Y-%m-%d %H:%M")
                    cm.append({"id_user":tweet.user.id_str, "name_user":tweet.user.screen_name, "time_comment":time, \
                    "id_comment":tweet.id_str, "text_comment" : tweet.full_text, "id_tweet" : tweet_id})
    return cm

def rm_main(n, persist=True):
    cm = list()
    start = datetime.now()
    print("it runs!")
    print("\nStarting time:", start)

    IDs = fetchMysql(n)

    for i, id in enumerate(IDs):
        print("\n\n ***** Tweet No: {} ***** \n\n".format(i+1))
        #id_manuel = '1278678613324369920'
        cm = getcomments(id)
        
        #Insert Obtained Comments
        insertMysql(cm) 

        print("\nNumber of the comments:", len(cm))
        
        with open('data-%s.json'%(id), 'w') as f:
            json.dump(cm, f)        
    
    finish = datetime.now()
    print("--->Finishing time", finish)
    difference = finish - start
    print("Total time passed:", difference)
    print("Total {} minutes:".format(difference.total_seconds() / 60.0))
    
    return cm

if __name__ == '__main__':
    n = 10
    rm_main(n)
    
    
    
    