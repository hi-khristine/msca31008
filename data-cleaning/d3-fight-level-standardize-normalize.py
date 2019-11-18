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

X = pd.read_csv( '../out/d2-fight-level-handle-strings-dates.csv'  )

# extract the Winner column.
y = X.Winner
X.drop( 'Winner', axis = 1, inplace = True )

# drop problematic columns.
X.drop( ['Diff_draw', 'Unnamed: 0', 'Unnamed: 0.1' ], axis = 1, inplace = True )

dd = ddict(X)

X.isna().sum()[ X.isna().sum() > 0 ]

# standardize.
from sklearn.preprocessing import StandardScaler
cols = X.columns
X = StandardScaler().fit_transform(X)

# normalize.
from sklearn.preprocessing import Normalizer
X = Normalizer().fit_transform(X)

save( '../out/d3-fight-level-standardize-normalize.pkl', X, y, cols )

del dd, X, y, cols