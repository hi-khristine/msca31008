# read functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data. Change winner to binary 1/0:
load( '../out/d3-fight-level-transform.pkl' )

import random
random.seed( 115 )

# train test split.
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.25
)

# run grid search on parameters.    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
m = DecisionTreeClassifier()
readpriorgrid = False
if ( not readpriorgrid  )  or ( 'grid-decisiontree.pkl' not in os.listdir('../out/') ):
    grid_decisiontree = GridSearchCV(
        m, cv = 8, scoring = 'accuracy', n_jobs = -1, verbose = True,
        param_grid = dict(
            # wide ranges:
            #min_impurity_decrease = [ x/1000 for x in list( range( 1, 10, 1 ) ) ]
            #max_features = range( 10, 25, 2 ),
            min_samples_split = range( 25, 35, 5 ),
            max_depth = range( 5, 15),
            max_features = range( 10, 25, 2 ),
            min_impurity_decrease = [ x/1000 for x in list( range( 1, 5, 1 ) ) ]
        )
    )
    grid_decisiontree.fit( X_train, y_train )
    save( '../out/grid-decisiontree.pkl', grid_decisiontree )
else:
    load( '../out/grid-decisiontree.pkl' )
print(grid_decisiontree.best_params_)
print(grid_decisiontree.best_score_)
m = grid_decisiontree.best_estimator_

#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#t = TSNE( learning_rate = 100 ).fit_transform(grid)
#plt.figure(1,figsize=(20,20),dpi=72)
#plt.scatter( x = t[:,0], y = t[:,1], c = grid.score )
#plt.show()

# Fit decision tree.
m.fit( X_train, y_train )

# accuracy against train data (in-model).
from sklearn.metrics import f1_score, precision_score, recall_score
p = m.predict(X_train)
print({ 'fscore': f1_score( y_train, p ), 'precision': precision_score( y_train, p ), 'recall': recall_score( y_train, p ) })

# accuracy against test data.
p = m.predict(X_test)
print( { 'fscore': f1_score( y_test, p ), 'precision': precision_score( y_test, p ), 'recall': recall_score( y_test, p ) } )

# kfold cv
from sklearn.model_selection import cross_validate
kfold = cross_validate(
    m, X, y, 
    cv = 10, 
    scoring = [ 'precision', 'recall' ], 
    n_jobs = -1 
)
print( 
  'Mean precision: %s \nMean recall %s' % 
  ( kfold['test_precision'].mean(), kfold['test_recall'].mean() ) 
)
