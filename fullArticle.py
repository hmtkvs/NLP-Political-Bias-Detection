from selenium import webdriver
import selenium.webdriver.support.ui as ui
import time
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd
import csv
import os
from datetime import date
import sys
import numpy as np

from tweet_cleaner import TweetCleaner
import for_server_predict as fsp

from webdriver_manager.chrome import ChromeDriverManager


def crawl(driver, tweet_id):
    
    text_list, title_list, author_list = [],[],[]

    for job_elem1, job_elem2, job_elem3  in zip(
        driver.find_elements_by_xpath('/html/body/div[5]/article/div[1]/h1'),
        driver.find_elements_by_xpath('//*[@id="body-text"]/div[1]'),
        driver.find_elements_by_xpath('/html/body/div[5]/article/div[1]/div[2]/div[1]/p[1]/span')):
        title_list.append(job_elem1)
        text_list.append(job_elem2.text)
        author_list.append(job_elem3.text())
    
    #df.at[0, "full_text"] = title_list[0] + text_list[0]      
    #df.at[0, "author"] = author_list[0]
    #df.at[0, "tweet_id"] = tweet_id
    print("title!!!",title_list)
    text = title_list[0] + text_list[0]
    
    print("done!", text)
    
    return text, author_list[0], tweet_id
    
def get_article(url):
    # importing the necessary packages
    import requests
    from bs4 import BeautifulSoup
    import numpy as np
    
    try:
        r1 = requests.get(url)
        coverpage = r1.content
        soup1 = BeautifulSoup(coverpage, 'html5lib')
    except:
        pass
    
    
    # Getting author's name
    try:
        author_name = soup1.find_all('span', class_='metadata__byline__author')[0].get_text()
    except:
        author_name = np.nan

    list_paragraphs = []

    # Getting the title
    coverpage_news = soup1.find_all('h1', class_='pg-headline')
    
    try:
        print("title:", coverpage_news[0].get_text())
        list_paragraphs.append(coverpage_news[0].get_text())
           
        # Reading the content (it is divided in paragraphs)
        article = requests.get(url)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')

        #Paragraphs0
        body = soup_article.find_all('div', class_='el__leafmedia el__leafmedia--sourced-paragraph')

        # Unifying the paragraphs
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        #Paragraphs1
        body = soup_article.find_all('div', class_='zn-body__paragraph speakable')

        # Unifying the paragraphs
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        #Paragraphs2
        body = soup_article.find_all('div', class_='zn-body__paragraph')

        # Unifying the paragraphs
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
        
        return final_article, author_name
    
    except:
        return np.nan, np.nan

    
    
    
def main():
    
    df = pd.DataFrame(columns = ["full_text", "author", "tweet_id"])
    
    text_list, author_list, id_list = [],[],[]
    #news_link, tweet_id = getLink()
    news_link, tweet_id = fsp.updateFullArticle('', 'fetch') 
    print(len(tweet_id), "articles have been found!")
    print(len(news_link), "links have been found!")
    
    for url, id in zip(news_link,tweet_id):
        #print(url, id)
        if not url[0:3] == "pic":
            full_text, author_name = get_article(url)
            if full_text is not np.nan:
                text_list.append(full_text)
                author_list.append(author_name)
                id_list.append(id)
                
    df["full_text"] = np.array(text_list)
    df["author"] =  np.array(author_list)
    df["tweet_id"] = np.array(id_list)
    df.to_csv("fullarticle22.csv")
    print(df.head())
    
    
    fsp.updateFullArticle(df,'insert')   
    #fsp.updateFullArticle(tweet_id,'delete')  
        
    
if __name__ == '__main__':
    main()
 
    
    