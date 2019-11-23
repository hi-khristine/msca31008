# author: Bryce C
# purpose: address NA issues.
# note: 
#   this code should be run from the data-cleaning directory.
#   when it runs, you'll have a corrected CSV file in the out/ folder
#   with the same name as this file.

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

import pandas as pd
import numpy as np

d = pd.read_csv('../out/a-fix-names.csv')

# check NAs before: 95390 NAs on 1942 rows
print( '%s NAs on %s rows' % (
    d.isna().sum().sum(),
    d.isnull().any(axis=1).sum()
))

# handle fighter metrics: these shouldn't change much.
# let's fill with the most recent prior measurement for each fighter.
# edit: after attempting this, there aren't any NAs that can be filled this way.
# so I've disabled it to speed up the code.
if False:
    
    # identify these measures.
    measures = [ 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'Stance' ]
    measuresbr = [ grep( '_' + i + '$', d.columns, value = True ) for i in measures ]
    measuresbr = [y for x in measuresbr for y in x]
    
    # now get them for each fighter.
    f = None
    for color in ['R','B']:
        cols = ['fighter'] + measures
        fc = d[ ['fightid', 'date'] + [ color + '_' + x for x in cols  ] ]
        fc.columns = ['fightid', 'date'] + cols
        fc.assign( color = color, inplace = True )
        f = pd.concat([f,fc]).reset_index(drop=True)
        del color, fc, cols
    
    # start with 1532 NAs.
    f.isna().sum().sum()
    
    # attempt fill.
    f.sort_values( [ 'fighter', 'date' ], inplace = True )
    for meas in measures:
        f[ meas ] = f.groupby( 'fighter' )[ meas ].transform(
            lambda s: np.nan if pd.isnull(s).all() == True else s.loc[ s.first_valid_index() ]
        )
        del meas
    
    # still 1532 NAs!
    f.isna().sum().sum()
    
    # code to explore a fighter with NAs:    
    f[ f.fighter == f.fighter[ f['Height_cms'].isna() ].values[0] ]

# 95390 NAs on 1942 rows
print( '%s NAs on %s rows' % (
    d.isna().sum().sum(),
    d.isnull().any(axis=1).sum()
))

# 3/4 fighters fight with the Orthodox stance, given 'data.csv' stance column.
pd.concat( [d.R_Stance, d.B_Stance ] ).value_counts( ascending = False )
pd.concat( [d.R_Stance, d.B_Stance ] ).shape
d.R_Stance.fillna( 'Orthodox', inplace = True )
d.B_Stance.fillna( 'Orthodox', inplace = True )

# 9 missing referee. Set to (Missing).
d.Referee.isna().sum()
d.Referee.fillna( '(Missing)', inplace = True )

# 95074 NAs on 1789 rows
print( '%s NAs on %s rows' % (
    d.isna().sum().sum(),
    d.isnull().any(axis=1).sum()
))

# can we fill in height, reach, weight by weight class?
# plot each by weight class. there is lots of variation.
import matplotlib.pyplot as plt
import seaborn as sns

# identify these measures.
measures = [ 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs' ]

# get them for each fighter.
f = None
for color in ['R','B']:
    cols = ['fighter'] + measures
    fc = d[ [ 'fightid', 'date', 'weight_class' ] + [ color + '_' + x for x in cols  ] ].reset_index(drop=True)
    fc.columns = [ 'fightid', 'date', 'weight_class'] + cols
    fc['color'] = color
    f = pd.concat([f,fc]).reset_index(drop=True)
    del color, fc, cols
    
sns.boxplot( x = f.Height_cms, y = f.weight_class )
sns.boxplot( x = f.Reach_cms, y = f.weight_class )
sns.boxplot( x = f.Weight_lbs, y = f.weight_class )
sns.boxplot( x = f.age, y = f.weight_class )

# there is a lot of overlap, but let's attempt to fill NAs physical measurement with the mean
# and see how that plays out.
for wc in d.weight_class.unique():
    for meas in measures:
        f[meas].loc[ f[meas].isna() & ( f.weight_class == wc ) ] = f[meas][ ~f[meas].isna() & ( f.weight_class == wc ) ].mean()
        del meas
    del wc
    
# confirm no NA.
f[measures].isna().sum()

# join this back to d. 
# note shape to make sure it doesn't change.
# (5144, 146)
d.shape

for color in ['R', 'B']:
    colormeas =  [ color + '_' + x for x in measures ]
    d.drop( colormeas, axis = 1, inplace = True )
    fc = f[ f.color == color ][ [ 'fighter', 'fightid' ] + measures ]
    fc.columns = [ color + '_fighter', 'fightid' ] + colormeas
    d = pd.merge( d, fc, how = 'left', on = [ color + '_fighter', 'fightid' ] )
    del color, colormeas, fc
    
# (5144, 146)
d.shape

# 93835 NAs on 1496 rows
print( 'Raw data has %s NAs on %s rows' % (
    d.isna().sum().sum(),
    d.isnull().any(axis=1).sum()
))

# drop first fights.
d = d[ 
      # no NAs in the row OR
      ~d.isnull().any(axis=1) | 
      # is not a first fight.
      ( d.R_fighter.duplicated() & d.B_fighter.duplicated() ) 
].reset_index( drop = True )

# 44051 NAs on 720 rows
print( 'Raw data has %s NAs on %s rows' % (
    d.isna().sum().sum(),
    d.isnull().any(axis=1).sum()
))

# remaining NAs are fight metrics. 
# leave them in for now, we'll use kmeans to fill them in once everything is standardized/normalized.
nacols = d.columns[ d.isna().sum() > 0 ]

d.to_csv( '../out/b-handle-nas.csv', index = False )

# clean workspace to prep for the next file to run.
del d, f, measures