# author: Bryce C
# purpose: standardize and normalize fight-level data for easier modeling clustering.
# note: 
#   this code should be run from the data-cleaning directory.
#   when it runs, you'll have a corrected CSV file in the out/ folder
#   with the same name as this file.

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

X = pd.read_csv( '../out/d1-fight-level-handle-strings-dates.csv' )

# calculate the winnings of calling a fight correctly.
X['winmult'] = [1] * X.shape[0]
if 'B_odds' in X.columns:    
    for i in X.index[ ~X.R_odds.isna() & ~X.B_odds.isna() ]:        
        winnerodds = X.R_odds.iloc[i] if X.Winner.iloc[i] == 1 else X.B_odds.iloc[i]
        X['winmult'].iloc[i] = winnerodds/100 + 1 if winnerodds > 0 else -100/winnerodds + 1
        del winnerodds, i

# extract the Winner and winnings columns.
y = X.Winner
winmults = X.winmult
X.drop( [ 'Winner', 'winmult' ], axis = 1, inplace = True )

# drop id column.
X.drop( [ 'fightid' ], axis = 1, inplace = True )

# standardize.
from sklearn.preprocessing import StandardScaler
cols = X.columns
X = StandardScaler().fit_transform(X)

# Normalizer won't work with NAs, so this is a good time to fill them in.

# if this is after iteration 0, use kmeans clustering to fill them.
if ( 'iterationnum' not in globals() ) or (iterationnum >= 1 ):
    
    labels, centroids, X = kmeans_missing( X, n_clusters = 20, max_iter = 10 )
    del labels, centroids

# if not, drop them.    
else: 
    
    X = pd.DataFrame(X)
    nas = X.isnull().any( axis = 1 )
    X = X[ ~nas ]
    y = y[ ~nas ]
    winmults = winmults[ ~nas ]
    del nas

# normalize.
from sklearn.preprocessing import Normalizer
X = Normalizer().fit_transform(X)

save( '../out/d2-fight-level-standardize-normalize-kmeansNA.pkl', X, y, cols, winmults )

del X, y, cols #, labels, centroids