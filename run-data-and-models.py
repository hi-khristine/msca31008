# author: Bryce C
# purpose: run all scripts in data-cleaning, then run models.
# note: 
#   this code should be run from the msca31008 directory.
#   when it finishes, processed files will be saved in the out/ directory.

# this file errors out and I'm not sure why.

# read in functions.
import os, sys, fnmatch, traceback, re, random

os.chdir( 'data-cleaning' )

# run each iteration.
allscores = None
kfold_splits = 10
for iterationnum in [0,1,2,3]:
#for iterationnum in [2,3]:
    
    # start with same random seed for each iteration.
    random.seed( 141 )
    
    iteration = [ 'A: Base Model', 'B: Fill NAs by KMeans, Weight Class', 'C: Add Odds', 'D: Grid Seach Param. Tuning' ][iterationnum]
    
    # run data cleaning at selected level.
    if iterationnum < 3: 
        for datacleanfile in fnmatch.filter( os.listdir('.'), '*.py' ): 
        
            # skip fighter-level, we aren't modeling it.
            if re.search( 'fighter-level', datacleanfile ):
                continue
            
            print(datacleanfile)
            try: 
                runfile(datacleanfile, wdir = os.getcwd())
            except:
                #einfo = sys.exc_info()
                #traceback.print_last( einfo )
                raise Exception( "Error at file [ " + datacleanfile + " ]." )
            del datacleanfile
    
    load( '../out/d3-fight-level-transform.pkl' )
    X = pd.DataFrame(X) 
    X.columns = cols
    X.reset_index( drop = True, inplace = True )
    y.reset_index( drop = True, inplace = True )
    winmults.reset_index( drop = True, inplace = True )
    print( X.shape )
    print( y.shape )
    
    # train test split.
    from sklearn.model_selection import train_test_split
    X_train , X_test, y_train, y_test, winmults_train, winmults_test = train_test_split(
        X, y, winmults,
        test_size = 0.25
    )
    
    # run models.
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
    from sklearn.model_selection import KFold
    from statistics import mean 
    
    # load tuned parameters. run the applicable .py script in models/ to get these.
    if iterationnum >= 3:
        
        load( '../out/grid-randomforest.pkl' )
        load( '../out/grid-adaboost.pkl' )
        load( '../out/grid-decisiontree.pkl' )
        load( '../out/grid-svc.pkl' )
        
        models = [
            ('Predict Red', PredictRed() ),
            ('Predict Higher Odds', PredictHighOdds() ),
            ('Logistic Regression', LogisticRegression( solver = 'lbfgs', n_jobs = -1 )),
            ('ADA Booster', grid_adaboost.best_estimator_ ),
            ('Decision Tree', grid_decisiontree.best_estimator_ ),
            ('Random Forest', grid_randomforest.best_estimator_ ),
            ('SVC', grid_svc.best_estimator_ )
        ]
        
    else:        
    
        models = [
            ('Predict Red', PredictRed() ),
            ('Predict Higher Odds', PredictHighOdds() ),
            ('Logistic Regression', LogisticRegression( solver = 'lbfgs', n_jobs = -1 )),
            ('ADA Booster', AdaBoostClassifier()),
            ('Decision Tree', DecisionTreeClassifier( 
                max_depth = 20, 
                min_samples_split = .03
            )),
            ('Random Forest', RandomForestClassifier( 
                n_estimators = 200, 
                min_samples_leaf = .1,
                n_jobs = -1
            )),
            ('SVC', SVC( kernel='linear' ))
        ]
    
    # svc causes an error in voting classifier, so only use the first 4 models.
    models.append( ('Voting Classifier', VotingClassifier(models[2:6]) ) )
    
    # identify columns we'll drop for log regression: 
    # find highly correlated pairs.
    # for each, return the lowest-correlated one to drop later.
    hicor = []
    todrop = []
    c = X_train.corr()
    for row in range(c.shape[0]):
        for col in range(c.shape[1]):
            if col >= row: continue
            if c.iloc[row,col] > .8: 
                hicor.append( [ cols[row], cols[col] ] )
                rowcor = np.corrcoef(X_train.iloc[:,row],y_train)[0][1]
                colcor = np.corrcoef(X_train.iloc[:,col],y_train)[0][1]
                if rowcor < colcor:
                    todrop.append( cols[row] )
                else:
                    todrop.append( cols[col])
                del rowcor, colcor
            del col
        del row
    del hicor, c
    
    for modelname, model in models:
        
        # can't run odds until ut is in the data.
        if ( iterationnum < 2 ) and ( modelname == 'Predict Higher Odds' ): continue
        
        print( 'Running: ' + modelname + ' / ' + iteration )
        
        idrop = todrop if modelname == 'Logistic Regression' else []
        
        # in-model (train):
        model.fit( X_train.drop( idrop, axis = 1 ), y_train )
        p = model.predict( X_train.drop( idrop, axis = 1 ) )
        
        scores = {
            'type': ['A: Train'],
            'fscore': [f1_score( y_train, p )],
            'precision': [precision_score( y_train, p )],
            'recall': [recall_score( y_train, p )],
            'accuracy': [accuracy_score( y_train, p )],
            'meanwinnings': [mean(
                [ winmults_train[ winmults_train.index[i] ] - 1 if y_train[ y_train.index[i] ] == p[i] else -1 for i in range(len(y_train)) ]
            )]
        }
        
        # test:
        p = model.predict( X_test.drop( idrop, axis = 1 ) )
        scores['type'].append('B: Test')
        scores['fscore'].append(f1_score( y_test, p ))
        scores['precision'].append(precision_score( y_test, p ))
        scores['recall'].append(recall_score( y_test, p ))
        scores['accuracy'].append(accuracy_score( y_test, p ))
        scores['meanwinnings'].append(mean(
            [ winmults_test[ winmults_test.index[i] ] - 1 if y_test[ y_test.index[i] ] == p[i] else -1 for i in range(len(y_test)) ]
        ))
        
        # cross-validation.
        cvmets = { 'fscore': [], 'precision': [], 'recall': [], 'accuracy': [], 'meanwinnings': [] }
        for train_index, test_index in KFold( n_splits = kfold_splits, shuffle = True ).split(X.drop( idrop, axis = 1 )):
            X_train_cv, X_test_cv = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train_cv, y_test_cv = y[train_index], y[test_index]
            winmults_train_cv, winmults_test_cv = winmults[train_index], winmults[test_index]
            model.fit( X_train_cv, y_train_cv )
            p_cv = model.predict( X_test_cv )
            cvmets['fscore'].append(f1_score( y_test_cv, p_cv ))
            cvmets['precision'].append(precision_score( y_test_cv, p_cv ))
            cvmets['recall'].append(recall_score( y_test_cv, p_cv ))
            cvmets['accuracy'].append(accuracy_score( y_test_cv, p_cv ))
            cvmets['meanwinnings'].append(mean(
                [ winmults_test_cv[ winmults_test_cv.index[i] ] - 1 if y_test_cv[ y_test_cv.index[i] ] == p_cv[i] else -1 for i in range(len(y_test_cv)) ]
            ))
            del X_train_cv, X_test_cv, y_test_cv, y_train_cv, p_cv, train_index, test_index, winmults_test_cv, winmults_train_cv
        scores['type'].append('C: Cross-Validation')
        for metricname, metricvals in cvmets.items():
            scores[ metricname ].append( np.mean( metricvals ) )
            del metricname, metricvals
        del cvmets  
        
        print( scores )
        
        allscores = pd.concat( [allscores, pd.DataFrame(scores).assign( 
                model = modelname, 
                iteration = iteration, 
                iterationnum = iterationnum,
                rows = X.shape[0],
                cols = X.shape[1],
                kfold_splits = kfold_splits
        ) ] ).reset_index( drop = True )
        
        del modelname, model, scores, p, idrop
        
    del X, y, cols, iteration, iterationnum, todrop, y_test, y_train, X_test, X_train, models
        
allscores.to_csv( '../out/allscores.csv', index = False )  
# del allscores