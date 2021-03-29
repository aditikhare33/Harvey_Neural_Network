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
        
        
    Using Linear Regression to predict
        predict what the column vectors
        will return
        
        absolute_value(coef) closer to 0 = that column tag less correlated to that feature
        coef closer to 1
        
    Gaussian continuous (like how OLS)
    Benoulli: Linear model
    Catgorical with ranking
    
    USE: multinomial, don't know ranking
"""

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

