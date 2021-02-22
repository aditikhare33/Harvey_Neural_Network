"""Harvey Classifier"""

import csv #to parse csv files (data is in csv fomat)

import pathlib #to create paths for files, this can be helpful if we want to create any files and place them somewhere

#import json
#import os

import sys
import numpy
#stops array for DTM from being truncated:
numpy.set_printoptions(threshold=sys.maxsize)

import sklearn
from sklearn.feature_extraction.text import CountVectorizer #for creating DTM (Document Term Matrix)
from sklearn import linear_model

import stop_words #for specific words we dont need in DTM

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import click #command line


#TODO: copied this class from scikit
#TODO: figure out how to do lemmatizing and stemming within sklearn?
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def load_data(training_file, verbose, mode = 0):
    if mode == 0: #make all of text column into a large list
       
        with open(training_file, 'r') as csv_file:
            lines = csv_file.readlines()
        
        tweets = list() #equivalent to []
        
        for line in lines:
            data = line.split(',')
            tweets.append(data[1])
         
        if verbose:
            print("printing tweets:\n", tweets)
        
        return tweets
            
    elif mode == 1:
        print("converting tweets into singular words (json style)")
        return dict()
    elif mode == 2:
        print("account for thee label a tweet has in its parsing")
        return list()
        

def feature_matrix(tweets, verbose):
   #stemming: group together words grammatically (like: run, ran, running)
   ps = PorterStemmer()
   corpus = tweets
   #corpus = ps.stem(tweets)
   
   #stop words to get rid of; common words like "that, there"
   my_stop_words = stop_words.get_stop_words('english')
   my_stop_words += ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

   
   print("\n\n\nPrinting stop words\n", my_stop_words)
   
    
   #DTM function from scikit
   vectorizer = CountVectorizer(stop_words = my_stop_words,
                                strip_accents = 'ascii',
                                max_features = 250,
                                #tokenizer = ps.stem()
                                #tokenizer = LemmaTokenizer() <-- throws an error
                                )
    #^^strip_accents = 'ascii' gets rid of the weird symbols in some of the text
       
   X = vectorizer.fit_transform(corpus)
   

   print("\n\n\nPrinting feature_names")
   print(vectorizer.get_feature_names())
   print("printing feature as an array (to output file)\n")
   #print(X.toarray())
   
   return X.toarray()
   

def ordinary_least_squares_regression():
    print("haha")

def categorical_naive_bayes():
    print("haha")
    

    
"""
Binary variable: ex: politics yes or None
    feature matrix (some susbet of term frequency matrix)
    then use feature matrix in OLS
        is it about topic
            is it positive affect about topic
            is it negative affect about topic
        is it not about topic
        
        make DTM (see screenshot)
        drop stop words (like and, for, common words)
        can do "stemming" (take common conjugations (run, running, ran into one column)
        
        
    after DTM: run regression
        regression fits binary class to feature matrix
        
        pnorm() <-- take continuous vairable turn into non-continous
        
        cross features
        
    DTM matrix with just the text
        then add in tags later because lines probably in same order
        
        SO take many many featuryes --> make into one
        
    Linear regression Model
        6 topics --> create 6 binary variables
        column vectors for each  (6 columns, n tweets rows, nbig array)
        
        regression line: take one of the columns in the array and regress by DTM
"""
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
                       "(3) Quit\n")
    #import pdb; pdb.set_trace()
    if userInput == '0':
        print("building feature matrix")
        array = feature_matrix(load_data(training_file, verbose), verbose)
        
        filestream = open(str(training_file) + "DTM_output_file.txt", "w+")
        filestream.write(str(array))
        filestream.close()
        
    elif userInput == '1':
        print("starting ordinary least squares model")
        load_data()
        ordinary_least_squares_regression()
    elif userInput == '2':
        print("starting categorical naive bayes")
        load_data()
        categorical_naive_bayes()
    else:
        print("exiting")
        exit()
        
# This is how python tells if the file is being run as main
if __name__ == '__main__':
    main()
    
#cateogorical naive bayes

#ordinary least squares

#logicistic regression
    #can make a binary variable for each like 1 protest or not protest
    
#feature matrixes (unigram dictionary but then figure out which words are most common
    #pytorch
    
#Fariss other office hours:Zoom Office Hours: Monday 2:00pm-4:00pm,
# Zoom Office Hours: Tuesday, 10:00am-2:00pm
# Zoom Office Hours: Wednesday, 5:30pm-6:30pm
# and by appointment.



# human rights reports mnay organizaitons from 1970s
    # most every country in the world, 1 or more reports for every country every year
    
#html v. xml
    #xml not just for websitres, for excel too and other platforms
