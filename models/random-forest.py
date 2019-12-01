import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data. Change winner to binary 1/0:
load( '../out/d3-fight-level-transform.pkl' )
print( X.shape )

# train test split.
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state = 718,
    test_size = 0.25
)

# run grid search on parameters.    
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
m = RandomForestClassifier()
readpriorgrid = False
if ( not readpriorgrid  )  or ( 'grid-randomforest.pkl' not in os.listdir('../out/') ):
    grid_randomforest = GridSearchCV(
        m, cv = 6, scoring = 'precision', n_jobs = -1, verbose = True,
        param_grid = dict(
            max_depth = range( 8, 16, 1 ),
            min_samples_leaf = [ x/100 for x in range( 1, 10, 1 ) ],
            n_estimators = range( 1, 11, 1),
        )
    )
    grid_randomforest.fit( X_train, y_train )
    save( '../out/grid-randomforest.pkl', grid_randomforest )
else:
    load( '../out/grid-randomforest.pkl' )
print(grid_randomforest.best_params_)
print(grid_randomforest.best_score_)
m = grid_randomforest.best_estimator_

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
