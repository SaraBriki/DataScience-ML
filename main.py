# imports
import numpy as np
import pandas as pd

# read datasets
data_imdb = pd.read_csv("imdb_labelled.txt", delimiter="\t", header=None)
data_imdb.columns = ["Review_text", "Review class"]

data_amazon = pd.read_csv("amazon_cells_labelled.txt", delimiter="\t", header=None)
data_amazon.columns = ["Review_text", "Review class"]

data_yelp = pd.read_csv("yelp_labelled.txt", delimiter="\t", header=None)
data_yelp.columns = ["Review_text", "Review class"]

data = pd.concat([data_imdb, data_amazon, data_yelp])

# output concatenated portions of datasets

data

# important libraries
import re
import nltk
import string
from nltk.tokenize import word_tokenize, punkt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# clean_text fucntion

def clean_text(dataframe):
    all_reviews = list()  # corpus
    lines = dataframe["Review_text"].values.tolist()
    for text in lines:
        text = text.lower()

        # removing links
        pattern = re.compile('http[s]?://(:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)

        # removing symbols
        text = re.sub(r"[,.\"!@#$%^&*-+=<>~:;`(){}?/;`~:<>+=-]", "", text)

        # tokenizing text, returns list rather than string returned by the split() function
        tokens = word_tokenize(text)

        # removing punctuation
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # ignore everything that is not an alphabet
        words = [word for word in stripped if word.isalpha()]

        # removing some stopwords, very frequently used in language
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")

        # stemming
        PS = PorterStemmer()
        words = [PS.stem(w) for w in words if not w in stop_words]

        # constructing & returning list
        words = ' '.join(words)
        all_reviews.append(words)
        return all_reviews
    from sklearn.model_selection import train_test_split

    # using an 80% 20% split with random state 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # import class corresponding to model
    from sklearn.naive_bayes import GaussianNB

    # make an object out of it
    model = GaussianNB()

    # fit our training data in the model
    model.fit(X_train, y_train)

    # apply the model
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score, precision_score
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))


# cleaning our corpus and printing its first 20 documents
reviews = clean_text(data)
reviews[0:20]

# building vocabulary
from sklearn.feature_extraction.text import CountVectorizer

# making an object selecting all the words that occured at least 3 times in whole corpus then applying the object to our list of reviews
CV = CountVectorizer()
X = CV.fit_transform(reviews).toarray()
y = data["Review class"]

# TF-IDF Technique
# from sklearn.feature_extraction.text import TfidfVectorizer
# TV = TfidfVectorizer(min_df=3)
# X = TV.fit_transform(all_reviews).toarray()
# y = data.as_matrix(["Review_class"])
# Now if we print X[0] we won't get only binary values 0 or 1, we will get values of fractions that represent the product of TF and IDF of each feature in the document

print(X.shape)
print(y.shape)

