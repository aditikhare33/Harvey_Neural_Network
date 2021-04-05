"""Harvey Classifier"""

import csv #to parse csv files (data is in csv fomat)
import pathlib #to create paths for files, this can be helpful if we want to create any files and place them somewhere
import sys

import numpy
numpy.set_printoptions(threshold=sys.maxsize) #stops array for DTM from being truncated

import sklearn
from sklearn.feature_extraction.text import CountVectorizer #for creating DTM (Document Term Matrix)
from sklearn import linear_model
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

import stop_words #for specific words we dont need in DTM

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import click #command line

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def load_data(training_file, verbose, mode = 0):
    if mode == 0: #make all of text column into a large list
       #get tweets text
        with open(training_file, 'r') as csv_file:
            lines = csv_file.readlines()
        
        tweets = list() #equivalent to []
        
        line_count = 0
        for line in lines:
            if line_count > 0:
                data = line.split(',')
                if data[2].isdigit():
                    tweets.append(data[1])
            line_count += 1
        return tweets
            
    elif mode == 1:
        # get tweets as the labels given to them
        
        with open(training_file, 'r') as csv_file:
            lines = csv_file.readlines()
        
        tweets = list() #equivalent to []
        
        line_count = 0
        for line in lines:
            if line_count > 0:
                data = line.split(',')
                if data[2].isdigit():
                    tweets.append(int(data[2]))
            line_count += 1
        
        return tweets

def feature_matrix(tweets, verbose):
   #stemming: group together words grammatically (like: run, ran, running)
   ps = PorterStemmer()
   corpus = tweets
   #corpus = ps.stem(tweets)
   
   #stop words to get rid of; common words like "that, there"
   my_stop_words = stop_words.get_stop_words('english')
   my_stop_words += ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
   #DTM function from scikit
   vectorizer = CountVectorizer(stop_words = my_stop_words,
                                strip_accents = 'ascii',
                                max_features = 1000
                                #tokenizer = ps.stem()
                                #tokenizer = LemmaTokenizer() <-- throws an error
                                )
   #^^strip_accents = 'ascii' gets rid of the weird symbols in some of the text
   X = vectorizer.fit_transform(corpus)
   return X.toarray()
   
   
def column_vector(i, training_file):
    with open(training_file, 'r') as csv_file:
        lines = csv_file.readlines()
    
    column_vector = list() #equivalent to []
    data = list()
    
    
    line_num = 0;
    for line in lines:
        if line_num > 0:
            data = line.split(',')
            if data[2].isdigit() and int(data[2]) == i:
                column_vector.append(1)
            else:
                column_vector.append(0)
        line_num += 1
        
    # print("printing column vector for ", i, " :", column_vector)
    return column_vector
   

def ordinary_least_squares_regression(DTM_array, training_file, predicting_file):
    print("training the models with data")
    coef = list()
    # print("printing array: ", DTM_array)
    predictions = []
    new_DTM_array = []
    for i in range(1, 7):
        reg = linear_model.LinearRegression()
        fit = reg.fit(DTM_array, column_vector(i, training_file))
        new_DTM_array = feature_matrix(load_data(predicting_file, False), False)
        result = reg.predict(new_DTM_array)
        predictions.append(result)
        # print("printing topic ", i, "OLS output prediction:", result)
        coef.append(reg.coef_)
 
    model_predict_categorical(predictions, len(new_DTM_array), predicting_file)
        
        
# CHANGE CATEGORICAL to MULTINOMIAL
# BERNOULLI, 6 BINARY COLUMN VECTORS SEPARATELY (<--- should be similar to LINEAR LOGISTIC REGRESSION)
# MULTINOMIAL, 1 COLUMN VECTOR, probability for all 6 categories
#first thing to think about is how is the target measuesured

def categorical_naive_bayes(DTM_array, training_file, predicting_file):
    coef = list()
    
    predictions = []
    new_DTM_array = []
    for i in range(1, 7):
        clf = CategoricalNB()
        fit =  clf.fit(DTM_array, column_vector(i, training_file))
        new_DTM_array = feature_matrix(load_data(predicting_file, False), False)
        result = clf.predict(new_DTM_array)
        predictions.append(result)
        # print("printing topic ", i, "OLS output prediction:", result)
        coef.append(clf.coef_)

    model_predict_categorical(predictions, len(new_DTM_array), predicting_file)
    
def multinomial_naive_bayes(DTM_array, training_file, predicting_file):
    clf = MultinomialNB()
    clf.fit(DTM_array, load_data(training_file, False, 1))
    new_DTM_array = feature_matrix(load_data(predicting_file, False), False)
    predictions = clf.predict(new_DTM_array)
    real_values = load_data(predicting_file, False, 1)
    model_accuracy(len(new_DTM_array), real_values, predictions)
    
def bernoulli_naive_bayes(DTM_array, training_file, predicting_file):
    clf = BernoulliNB()
    clf.fit(DTM_array, load_data(training_file, False, 1))
    new_DTM_array = feature_matrix(load_data(predicting_file, False), False)
    predictions = clf.predict(new_DTM_array)
    real_values = load_data(predicting_file, False, 1)
    model_accuracy(len(new_DTM_array), real_values, predictions)
    
    
# OLS and categorical naive bayes predictions
def model_predict_categorical(predictions_pre, num_tweets, predicting_file):
    predictions = []

    for element in range(0, num_tweets):
        curr_prediction = 1
        curr_value = abs(predictions_pre[0][element])
        for i in range(1, 6):
            if abs(1 - abs(predictions_pre[i][element])) < abs(1 - curr_value):
                curr_prediction = i + 1
                curr_value = abs(predictions_pre[i][element])
                
        predictions.append(curr_prediction)

    real_values = load_data(predicting_file, False, 1)
    # print("printing real_values", real_values)
    
    model_accuracy(num_tweets, real_values, predictions)
    
def model_accuracy(num_tweets, real_values, predictions):
    print("beginning predictions")
    
    inacc_count = 0
    
    adjusted_num_tweets = num_tweets
    adjusted_inacc_count = 0
    
    num_clim_change = 0
    inacc_clim_change = 0
    for i in range(0, num_tweets):
        if real_values[i] != predictions[i]:
            inacc_count += 1
    
        if real_values[i] == 6:
            adjusted_num_tweets -= 1
        elif real_values[i] != predictions[i]:
            adjusted_inacc_count += 1
        
        if predictions[i] == 5 or predictions[i] == 1: #if tweet predicted to be abt climate change or Harvey
            num_clim_change += 1
            if (real_values[i] != 1 and real_values[i] != 5):
                inacc_clim_change += 1
            
    print("num_tweets on clim_change: ", num_clim_change)
    print("percent prediction tags (clim_change) correct =", 100 - ((inacc_clim_change/num_clim_change) * 100.0000000))
    
    print("\n")
    
    print("adjusted_num_tweets: ", adjusted_num_tweets)
    print("percent prediction tags (non_misc) correct =", 100 - ((adjusted_inacc_count/adjusted_num_tweets) * 100.0000000))
    
    print("\n")
    
    print("total num_tweets: ", num_tweets)
    print("percent prediction tags (non_misc) correct =", 100 - ((inacc_count/num_tweets) * 100.0000000))
    
    print("\n")
        

@click.command()
@click.argument('training_file', type=click.Path(exists=True))
@click.argument('predicting_file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True, help='Print more output.')
def main(training_file, predicting_file, verbose):
    """Harvey Classifier"""
    
    userInput =  input("Enter your choice:\n"
                       "(0) feature matrix\n"
                       "(1) ordinary least squares\n"
                       "(2) categorical naive bayes\n"
                       "(3) multinomial naive bayes\n"
                       "(4) bernoulli naive bayes"
                       "(5) exit\n")
    if userInput == '0':
        print("building feature matrix")
        array = feature_matrix(load_data(training_file, verbose), verbose)
        
        filestream = open(str(training_file) + "DTM_output_file.txt", "w+")
        filestream.write(str(array))
        filestream.close()
        
    elif userInput == '1':
        print("starting ordinary least squares model")
        verbose = False
        array = feature_matrix(load_data(training_file, verbose), verbose)
        ordinary_least_squares_regression(array, training_file, predicting_file)
    elif userInput == '2':
        print("starting categorical naive bayes")
        array = feature_matrix(load_data(training_file, verbose), verbose)
        categorical_naive_bayes(array, training_file, predicting_file)
    elif userInput == '3':
        print("starting multinomial naive bayes")
        array = feature_matrix(load_data(training_file, verbose), verbose)
        multinomial_naive_bayes(array, training_file, predicting_file)
    elif userInput == '4':
        print("starting bernoulli naive bayes")
        array = feature_matrix(load_data(training_file, verbose), verbose)
        bernoulli_naive_bayes(array, training_file, predicting_file)
    else:
        print("exiting")
        exit()
        
# This is how python tells if the file is being run as main
if __name__ == '__main__':
    main()
