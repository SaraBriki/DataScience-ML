# Bag-of-words

## Description
This is a mini-project I made when I first started learning NLP (Natural Language Processing), and it's a statistical language model based on word count that represents a text as the bag of the words composing it. It's a very simple representation and it was a first introductory step that helped me get more familiar with NLP.

## Contents

### Collected data
This project contains **.txt** and **.csv** files that correspond to the textual data we'll be processing.
We have three main data sources:
* _amazon_cells_labelled.txt_ and its respective **.csv** file.
* _imdb_labelled.txt_ and its respective **.csv** file.
* _yelp_labelled.txt_ and its respective **.csv** file.

These files contain labelled data, with the labels being either **0** or **1**, 1 for a positive statement and 0 for a negative one.

#### Description of the content of files
* The _amazon_cells_labelled.txt_ file contains customer feedbacks on a cellphone sold on Amazon.
* The _imdb_labelled.txt_, as it name suggests, are reviews on the _imdb_ site for film/cinema productions.
* The _yelp_labelled.txt_ contain customer reviews and recommendations on places (restaurants, bars, etc.) they had visited on _yelp_ website.

### Processing code
In the _main.py_ Python file lies all the code that I had written to parse, process and structure the available textual data into the BoW model.
In this file, we started by **importing the necessary librairies** and then we moved to the processing of data as follows:
1. Implementing a _clean_text()_ function to clean our textual data. It takes in a dataframe from which we remove symbols, links, punctuation, stopwords, etc. The text is tokenized as a list for easier manipulation of its contents. We subsequently perform a stemming of these words to reduce them to their original bases.
2. Performing an 80/20 train-test split on the processed data.
3. Importing the GaussianNB (Naive-Bayes) model in which we fit the training data.
4. Applying the model to make predictions and calculating performance metrics.

