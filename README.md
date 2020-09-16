
## Train
Trains CNN + LSTM hybrid model with BERT embeddings.
> python Train/train_words_dl_model.py -a 1 -w 1 -p "/homedtic/hkavas/SemEval" 

## Get Tweets
Obtains tweets for given username and timespan. This command runs periodically in our NiFi implementation.
> python Exporter.py --username "cnn" --since 2020-09-01 --until 2020-09-15

## Scrape Full Article
Checks news tweets whose author name has not been entered, uses link from database to scrape full article and the author.
> python fullArticle.py

## Get Comments
Checks database for new tweets and then finds replies to them.
> python getComments.py 

## Predict Bias
Checks database for comments or tweets whose bias has not been assigned. If there is any, it makes predictions and inserts into database.
> python dynamic_bias.py

# THE BIAS EFFECT OF NEWS MEDIA SOURCES ON SOCIAL MEDIA USERS
This repository contains the source code for detecting dynamic bias as it is described in our paper.
In this project, we aim to detect the possible effects of media bias by newspapers on public opinion. Our purpose is to create a system that dynamically detects bias in media and the comments.

## Installation
1. Clone this repository
2. Install dependencies
> pip install -r requirements.txt
3. For training:
  - cd Train
  - Download BERT embeddings from:
    > (https://github.com/google-research/bert)
  - Run the code
    > python train_words_dl_model.py -a 1 -w 1 -p "/homedtic/hkavas/SemEval"
    








