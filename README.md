
# THE BIAS EFFECT OF NEWS MEDIA SOURCES ON SOCIAL MEDIA USERS
This repository contains the source code for detecting dynamic bias as it is described in our paper.
In this project, we aim to detect the possible effects of media bias by newspapers on public opinion. Our purpose is to create a system that dynamically detects bias in media and the comments.

## Running the repository
Sections below demonstrates the requirements and steps to run the codes for each task.
### Installation
1. Clone this repository
2. Install dependencies
  > pip install -r requirements.txt
3. Connect to MySQL
  - Enter your user *name*, *password* and *database_name* into [MySQLsettings.json](https://github.com/hmtkvs/MastersThesis/blob/master/MySQLsettings.json)
  - Follow the steps in (https://dev.mysql.com/doc/connector-python/en/connector-python-installation.html) to download MySQL connector 
    > pip install mysql-connector-python
  - Open your MySQL Console
    > mysql> CREATE DATABASE database_name;
    > mysql> use database_name;
    > mysql> CREATE TABLE tweets (id_tweet VARCHAR(45) NOT NULL, text_tweet longtext,
       time_tweet VARCHAR(45) NOT NULL, link_tweet VARCHAR(100), news_link VARCHAR(100), predicted_bias VARCHAR(10), author_name VARCHAR(200), PRIMARY KEY (id_tweet))
       
    > mysql> CREATE TABLE comments (id_user VARCHAR(45) NOT NULL, name_user VARCHAR(45),
       text_comment VARCHAR(350), time_comment VARCHAR(100), id_comment VARCHAR(100), id_tweet VARCHAR(100), predicted_bias VARCHAR(10), PRIMARY KEY (id_user))
    * database should look like this when you have followed the steps above.
    ![image](https://user-images.githubusercontent.com/56079783/93386616-410ed600-f868-11ea-9d2e-c0e36d2df4db.png)
         
### Training
- `cd Train`
- Download BERT embeddings from:
  > (https://github.com/google-research/bert)
- `cd data/Articles`
- Add your data here(should be in xml format or could be replaced with other file formats by changing in [train_words_dl_model.py](https://github.com/hmtkvs/MastersThesis/blob/master/Train/train_words_dl_model.py)
- `cd ..`
- `cd ..`
- Use `-a` argument to train a different model:
  * `-a 1` hybrid CNN-LSTM model or
  * `-a 2` LSTM model.
- Use `-w` argument for embeddings
  * `-w 0` word2vec Embeddings
  * `-w 1` BERT Embeddings
- Run the code
  > python train_words_dl_model.py -a 1 -w 1 -p *yourpath/MastersThesis/Train*

* To evaluate the model, use -e option. This will run the script on evaluation mode, which loads the trained model from disk and runs it against the validation data to get  the model's evaluation metrics. The metrics will be printed in a log file.

### Get Tweets
This function retrieves tweets and insert them into Tweets table in MySQL.
- `cd MastersThesis`
- Select arguments for:
  * `--username` without using @
  * `--since` and `-until`. Example date format: `2020-09-01`
> python Exporter.py --username *username* --since *starting_date* --until *finishing_date* 

### Get Comments
Checks database for new tweets, finds replies to them and than inserts into MySQL.
> python getComments.py 

### Scrape Full Article
Checks news tweets whose author name has not been entered, uses link from database to scrape full article and the author.
* To run Google Chrome or Chromium should be installed.
  > python fullArticle.py
  
### Predict Bias
Checks database for comments or tweets whose bias have not been assigned. If there is any, this code makes predictions and inserts into database.
- Select arguments for:
  * `-p` the path that you are running the code
  * `-o` the directory that you want to put output
  * `-s` preferred sequence length
  * `-m` name of the model that will make the predictions
  -Run the code
    > python dynamic_bias.py -p *yourpath/MastersThesis/* -o yourpath/MastersThesis/results* -m "words_conv_lstm_model_w1_v1"

