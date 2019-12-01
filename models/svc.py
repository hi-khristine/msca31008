import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data. Change winner to binary 1/0:
load( '../out/d3-fight-level-transform.pkl' )
print( X.shape )

import random
random.seed( 1152 )

# train test split.
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.25
)

# run grid search on parameters.    
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
m = SVC()
readpriorgrid = False
if ( not readpriorgrid  )  or ( 'grid-svc.pkl' not in os.listdir('../out/') ):
    grid_svc = GridSearchCV(
        m, cv = 6, scoring = 'precision', n_jobs = -1, verbose = True,# error_score = 0.0,
        param_grid = dict(
            # these are for searching a wide range of values:
            C = [x/100 for x in range( 100, 150, 5 ) ],
            #kernel = [ 
            #    'linear', 'poly', 'rbf', 'sigmoid'
            #     precomputed results in error:  should be a square kernel matrix
            #    , 'precomputed' 
            #],
            #degree = range( 1, 3 ),
            #gamma = [ 'scale', 'auto' ],
            #coef0 = [x/100 for x in range( 0, 100 ) ],
            #shrinking = [ True, False ],
            #probability  = [ True, False ],
            #decision_function_shape = [ 'ovo', 'ovr' ],
            
            # these are narrowed down to the best.
            decision_function_shape  = ['ovo'],
            kernel = ['sigmoid'],
            probability = [True],
            shrinking = [False],
            gamma = ['scale'],
            degree = [1],            
        )
    )
    grid_svc.fit( X_train, y_train )
    save( '../out/grid-svc.pkl', grid_svc )
else:
    load( '../out/grid-svc.pkl' )
print(grid_svc.best_params_)
print(grid_svc.best_score_)
m = grid_svc.best_estimator_

#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#t = TSNE( learning_rate = 100 ).fit_transform(grid)
#plt.figure(1,figsize=(20,20),dpi=72)
#plt.scatter( x = t[:,0], y = t[:,1], c = grid.score )
#plt.show()

# Fit decision tree.
m.fit( X_train, y_train )

# accuracy against train data (in-model).
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( 
    y_train, 
    m.predict(X_train)
))

# accuracy against test data.
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( 
    y_test, 
    m.predict(X_test)
))
y_test.mean()

# check cross-validation.
crossval = False
if crossval: 
    from sklearn.model_selection import cross_validate
    kfold = cross_validate( 
        m, X_train, y_train, 
        cv = 10, 
        scoring = [ 'precision', 'recall' ], 
        n_jobs = -1 
    )
    print( 
      'Mean precision: %s \nMean recall %s' % 
      ( kfold['test_precision'].mean(), kfold['test_recall'].mean() ) 
    )
