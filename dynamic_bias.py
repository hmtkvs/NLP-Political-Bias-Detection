import pandas as pd
import os
import getopt
import sys
from time import sleep
import logging
from keras.models import load_model
from data_loaders import TokenizerLoader, TextSequencesLoader
from datetime import datetime

import mysql.connector
from mysql.connector import Error
from DBHandler import *
import Constants

from tweet_cleaner import TweetCleaner
import for_server_predict as fsp
                                    

class PredictBias:
    def __init__(self, sem_eval_path, runOutputFileName, model_file_name):
        self.sem_eval_path = sem_eval_path
        self.runOutputFileName = runOutputFileName
        self.model_file_name = model_file_name
 
    def _convert_texts_to_sequences(self, df):
        print(df.head())
        texts = df['text']
        
        logging.info('Loading tokenizer')
        tokenizer = TokenizerLoader(texts, sem_eval_path).load()
     
        logging.info('Converting texts to sequences')
        sequences_loader = TextSequencesLoader(tokenizer, seq_len)
        print("texts to seq\n", type(texts))
        print("texts to seq\n", texts)
        
        X_test = sequences_loader.load(texts.values)

        return X_test
    
    def _predict(self, model, X_val):

        logging.info('Predicting values')
        logging.info('X_val: ')
        logging.info(X_val)
        
        predicted_values = model.predict_classes(X_val)
        logging.info('predicted_values: ')
        logging.info(predicted_values)
        print('predicted_values: ', predicted_values)
        formatted_pred = predicted_values.reshape((-1,))
        return pd.Series(formatted_pred)
    
    def _create_output_dataframe(self, input_df, y_pred):
        logging.info('Creating output dataframe')

        truefalsedict = {0: False, 1: True}
        #y_pred_df = pd.DataFrame(y_pred, columns=['predicted_hyperpartisan'])
        input_df['predicted_bias'] = y_pred
        input_df['predicted_bias'] = input_df['predicted_bias'].map(truefalsedict, na_action='ignore')
        
        return input_df
        
    def main(self, data_type):
        df_final = pd.DataFrame()
                
        if data_type == 'tweet':
            #Get cleaner func.
            cleaner = TweetCleaner('', 'tweet') #-->Column name
        
            #predict tweet
            TWEEETs = fsp.fetchMysql('tweet')
            input_df = pd.DataFrame(TWEEETs, columns = ['id','tweet']) 
            df_tweet = cleaner.cleaning_table(input_df)
            df_tweet = df_tweet[pd.notnull(df_tweet['tweet'])]
            df_final = df_tweet.drop_duplicates(subset=['tweet'])
            
        elif data_type == 'comment':
            #Get cleaner func.
            cleaner = TweetCleaner('', 'tweet') #-->Column name
        
            #predict tweet
            COMMENTs = fsp.fetchMysql('comment')
            input_df = pd.DataFrame(COMMENTs, columns = ['id','tweet']) 
            df_comment = cleaner.cleaning_table(input_df)
            df_comment = df_comment[pd.notnull(df_comment['tweet'])]
            df_final = df_comment.drop_duplicates(subset=['tweet'])
                
        else:
            '''
            To make predictions from files. Change data kind in insertMysql before running
            '''
            #f= open("Results/DaggettBeaver.txt","r+")
            #fl = f.readlines()
            #data = list()
            #for x in fl:
            #    data.append(x.split())           
            #df = pd.DataFrame(data, columns = ['id','predicted_hyperpartisan']) 
            #df = df[pd.notnull(df['predicted_hyperpartisan'])]
        
            df_k = pd.read_csv("TheMEEye.csv", encoding= 'unicode_escape')
            #df_final = df_k [['text_tweet', 'id_tweet', 'predicted_bias']]
            df_final = df_k[pd.notnull(df_k['text'])]
            print(df_final.head())
                 
        #Get sequences
        X_test = self._convert_texts_to_sequences(df_final)
        print("This is how X_test looks like:", X_test[0:10])
        
        #Load model
        model = load_model(os.path.join(self.sem_eval_path, 'models', self.model_file_name))

        #Do the prediction
        y_pred = self._predict(model, X_test)
        
        # Create output dataframe to write on disk
        y_pred_df = self._create_output_dataframe(df_final, y_pred)
        print("This is how y_pred_df looks like:", y_pred_df.head())
        y_pred_df.to_csv(runOutputFileName, index = False)        
        
        fsp.insertMysql(y_pred_df, 'tweet')
        
        print("DONE!")
        

sem_eval_path = 'C:/Users/hmtkv/Desktop/THESSIS'
runOutputFileName = "PREDD.csv"
seq_len = 800
model_file_name = 'words_conv_lstm_model_w1_v111112.h5'

PredictBias(sem_eval_path, runOutputFileName, model_file_name).main('from_file')





    
    
    
