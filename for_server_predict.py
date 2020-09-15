import mysql.connector
from mysql.connector import Error
from DBHandler import *
import Constants

import sys

'''
TO CALL: 
    import for_server_predict as fsp
'''

def fetchMysql(data_type):
    COMMENTs = list()  
    TWEETs = list()  
    try:
        connection = DBHandler.get_mydb()
        cursor = connection.cursor() 
        
        if data_type == "tweet":
            #Count "tweets" table
            cursor.execute("select count(*) FROM tweets WHERE predicted_bias is NULL;")
            N = cursor.fetchall()
            count_tweets = N[0][0]
            print("Tweets are retrieved! No. of non-predicted tweets=", count_tweets)
            
            #Get data from "tweets" table    
            for i in range(0,count_tweets):
                cursor.execute("select id_tweet, text_tweet FROM tweets WHERE predicted_bias is NULL;") #LIMIT ADJUSTABLE OLMALI
                results = cursor.fetchall()
                TWEETs.append(results[i])
            print("Found TWEET Length:", len(TWEETs))         
            return TWEETs
        
        elif data_type == "comment":
            #Count "comments" table
            cursor.execute("select count(*) FROM comments WHERE predicted_bias is NULL;")
            N = cursor.fetchall()
            count_comments = N[0][0]
            print("Comments are retrieved! No. of non-predicted comments=", count_comments)

            #Get data from "comments" table    
            for i in range(0,count_comments):
                cursor.execute("select id_comment, text_comment FROM comments WHERE predicted_bias is NULL;") #LIMIT ADJUSTABLE OLMALI
                results = cursor.fetchall()
                COMMENTs.append(results[i])
            print("Found COMMENT Length:", len(COMMENTs))  
            return COMMENTs
            
    except Error as e:
        print("Error while connecting to MySQL", e)       
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            
    
def insertMysql(data, data_type):
    connection = DBHandler.get_mydb()
    cursor = connection.cursor() 
    try:
        print("This is the first id", data.at[0, 'id_tweet'])
        connection = DBHandler.get_mydb()
        cursor = connection.cursor() 
        
        data = data.drop_duplicates()
        #sys.exit()
        if data_type == "tweet":
            #Insert data into "tweets" table
            for i in range(0,len(data.columns)):
                print("DDATA IS HERE", data.head())
                id_text = data.at[i, 'id_tweet']
                bias_text = data.at[i, 'tweet_bias']
                print("ID SENT", id_text)
                cursor.execute("UPDATE tweets SET predicted_bias = (%s) WHERE id_tweet = (%s)",(str(bias_text), str(id_text)))
                connection.commit()       
            print("Bias inserted into tweets table!")
           
        elif data_type == "comment":   
            #Insert data into "comments" table
            for i in range(0,len(data)):
                id_text = data.iat[i, 0]#id
                bias_text = data.iat[i, 1]#predicted_bias
                print("id", id_text, "bias", bias_text)
                cursor.execute("UPDATE comments SET predicted_bias = (%s) WHERE id_comment = (%s)",(str(bias_text), str(id_text)))
                connection.commit()  

            print("Bias inserted into comments table!")

                                  
    except Error as e:
        print("Error while connecting to MySQL", e)       
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            
def updateFullArticle(data,  operation_type):
    news_link, id_tweet = [], []
    try:
        connection = DBHandler.get_mydb()
        cursor = connection.cursor() 
        print(operation_type)
        if operation_type == "fetch":
            cursor.execute("select news_link, id_tweet FROM tweets WHERE author_name is NULL;")
            results = cursor.fetchall()
            #print("RESULTS", results[0][1])
            for i in results:
                news_link.append(i[0].split()[0])
                id_tweet.append(i[1])
  
            return news_link, id_tweet
        
        elif operation_type == "insert":
                    
            #Insert data into "comments" table
            for i in range(0,len(data)):
                id_tweet = data.at[i, 'tweet_id']
                full_text = data.at[i, 'full_text']
                author_name = data.at[i, 'author']
                #
                try:
                    cursor.execute("UPDATE tweets SET text_tweet = (%s), author_name = (%s) WHERE id_tweet = (%s)",(full_text, author_name, str(id_tweet)))
                    connection.commit()  
                except:
                    print("id", id_tweet, "author name", author_name)
                    cursor.execute("UPDATE tweets SET text_tweet = (%s), author_name = (%s) WHERE id_tweet = (%s)",(full_text, '', str(id_tweet)))
                    connection.commit()
            print("Full text inserted into tweets table!")
        
        elif operation_type == "delete":   
            for i in range(0,len(data)):
                id_tweet = data[i]
                #cursor.execute("select * FROM tweets WHERE id_tweet = (%s)",(id_tweet))
                #results = cursor.fetchall()
                #print(results)
                #sys.exit()
                cursor.execute("DELETE FROM tweets WHERE id_tweet = (%s)",(id_tweet))
                connection.commit()
                
            
    except Error as e:
        print("Error while connecting to MySQL", e)       
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")    
            











