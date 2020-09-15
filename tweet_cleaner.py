#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm


import re

import os

import emoji

import pandas as pd
import numpy as np
import sys

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import (
    wordnet,
    stopwords
)

class TweetCleaner:
    def __init__(self, file_path, column_name):
        self.file_path = file_path
        self.column_name = column_name
        

    #preprocess text in tweets by removing links, @UserNames, blank spaces, etc.
    def preprocessing_text(self, table):
        print("Column name to be cleaned:",self.column_name) 
        table[self.column_name] = table[self.column_name].astype('str')
        
        #Delete not Latin characters
        encoded_string = table[self.column_name].apply(lambda x: x.encode("ascii", "ignore"))
        table[self.column_name] = encoded_string.apply(lambda x: x.decode())
        #put everythin in lowercase
        #table[self.column_name] = table[self.column_name].str.lower()
        #Replace rt indicating that was a retweet
        table[self.column_name] = table[self.column_name].str.replace('RT', '')
        #Replace occurences of mentioning @UserNames
        table[self.column_name] = table[self.column_name].replace(r'@\w+', '', regex=True)
        #Replace links contained in the tweet
        table[self.column_name] = table[self.column_name].replace(r'http\S+', '', regex=True)
        table[self.column_name] = table[self.column_name].replace(r'www.[^ ]+', '', regex=True)
        #remove numbers
        #table[self.column_name] = table[self.column_name].replace(r'[0-9]+', '', regex=True)
        #replace special characters and puntuation marks(keep exclamation and question mark)
        table[self.column_name] = table[self.column_name].replace(r'["$%&()*+,-./:;<=>@[\]^_`{|}~]', '.', regex=True)
        #replace #
        table[self.column_name] = table[self.column_name].replace(r'#', '', regex=True)
        #Replace emojis
        table[self.column_name] = table[self.column_name].replace(r'ud+', '', regex=True)
        
        return table


    #Replace elongated words by identifying those repeated characters and then remove them and compare the new word with the english lexicon
    def in_dict(self, word):
        if wordnet.synsets(word):
            #if the word is in the dictionary, it returns True
            return True

    def replace_elongated_word(self, word):
        regex = r'(\w*)(\w+)\2(\w*)'
        repl = r'\1\2\3'    
        if self.in_dict(word):
            return word
        new_word = re.sub(regex, repl, word)
        if new_word != word:
            return self.replace_elongated_word(new_word)
        else:
            return new_word

    def detect_elongated_words(self, row):
        regexrep = r'(\w*)(\w+)(\2)(\w*)'
        words = [''.join(i) for i in re.findall(regexrep, row)]
        for word in words:
            if not self.in_dict(word):
                row = re.sub(word, self.replace_elongated_word(word), row)
        return row

    def remove_emoji(self, string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        
        return emoji_pattern.sub(r'', string)

    def cleaning_table(self, table):
        #This function will process all the required cleaning for the text in our tweets
        table = self.preprocessing_text(table)
        table[self.column_name] = table[self.column_name].apply(lambda x: self.detect_elongated_words(x))
        table[self.column_name] = table[self.column_name].apply(lambda x: self.remove_emoji(x))
        return table

'''
CLEANING data from a file     

#def data_loader(self):
#    df_10k = pd.read_csv(self.file_path, sep=";", encoding= 'unicode_escape')
#    df_10k = df_10k.drop(['sep', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
#    df = df_10k[df_10k['permalink'].notnull()]
#    
#    df['text2'] = df['text'].apply(lambda x: x.split("pic.twitter.com")[0])
#    del df['text']
#    df['permalink2'] = df['permalink'].apply(lambda x: x.split("/")[-1])
#    del df['permalink']
#    
#
#    df = df.rename({'text2': 'tweet', 'permalink2': 'id_tweet'}, axis='columns')
#
#    print(df.iloc[0:3,0:3])
#
#    
#    return df

# CALL:
#cleaner = TweetCleaner(file_path)
#df = cleaner.data_loader()
#df_final = cleaner.cleaning_table(df)
#print(df_final['tweet'][44])
#df_final.to_csv("clean_10k.csv")
'''



