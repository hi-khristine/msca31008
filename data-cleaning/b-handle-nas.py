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

# for figher metrics, let's fill with the most recent prior measurement.

# start with original NA count. Range from 666 to 3.
fmets = [ 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'Stance' ]
fmetsbr = [ grep( '_' + i + '$', d.columns, value = True ) for i in fmets ]
fmetsbr = [y for x in fmetsbr for y in x]
f = pd.read_csv( '../data/fighter_level_dataset.csv' ).sort_values( by = [ 'fighter', 'date' ] )
f[fmets].isna().sum()

# attempt fill.
for i in fmets:
    f[ i ] = f.groupby( 'fighter' )[ i ].transform(
        lambda s: np.nan if pd.isnull(s).all() == True else s.loc[ s.first_valid_index() ]
    )
    del i
    
# we can fill Stance with missing. 3/4 fighters fight with the Orthodox stance, given 'data.csv' stance column.
f.Stance.loc[ f.Stance.isna() ] = 'Orthodox'

# check NA count again. it filled in some but lots of NA still.
f[fmets].isna().sum()

# plot height by weight class. there is lots of variation.
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot( x = f.Height_cms, y = f.weight_class )
sns.boxplot( x = f.Reach_cms, y = f.weight_class )
sns.boxplot( x = f.Weight_lbs, y = f.weight_class )

# there is a lot of overlap, but let's attempt to fill NAs physical measurement with the mean
# and see how that plays out.
for wc in f.weight_class.unique():
    for meas in [ 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs' ]:
        f[meas][ f[meas].isna() & ( f.weight_class == wc ) ] = f[meas][ ~f[meas].isna() & ( f.weight_class == wc ) ].mean()
        del meas
    del wc
    
# check NA count again. there are no nas in the fighter data now.
f[fmets].isna().sum()

# explore NA data.
# fna = f[ np.isin( f.fighter, f.fighter[ f.age.isna() ] ) ][ [ 'fighter', 'date' ] + fmets ]

# 35% of fighters have missing physical measurements.
# this is a large percentage but these will be difficult to address so let's drop them for now.
#nonafighters = f[fmets+['fighter']].dropna().fighter
#( f.fighter.nunique() - nonafighters.nunique() ) / f.fighter.nunique()
#d = d[ np.isin( d.R_fighter, nonafighters ) & np.isin( d.B_fighter, nonafighters ) ].reset_index( drop = True )

# attempt fill with first valid. 
for i in fmets:
    for br in [ 'B_', 'R_' ]:
        d[ br + i ] = d.groupby( br + 'fighter' )[ br + i ].transform(
            lambda s: np.nan if pd.isnull(s).all() == True else s.loc[ s.first_valid_index() ]
        )
        del br
    del i
    
# apply the same na approach as with fighters. 
# it would be better to use f but it'd take longer.
d.R_Stance.loc[ d.R_Stance.isna() ] = 'Orthodox'
d.B_Stance.loc[ d.B_Stance.isna() ] = 'Orthodox'
for wc in f.weight_class.unique():
    for meas in [ 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs' ]:
        for color in ['B_', 'R_']:
            meas2 = color + meas
            d[meas2][ d[meas2].isna() & ( d.weight_class == wc ) ] = d[meas2][ ~d[meas2].isna() & ( d.weight_class == wc ) ].mean()
            del color, meas2
        del meas
    del wc

# I am guessing NAs come from fights being first fights where no history exists.
d.sort_values( by = [ 'date' ], inplace = True )

# let's look at red: 49 columns have NAs.
dr = d[ grep( '^R_', d.columns, value = True ) ]
dr.isna().sum()[ dr.isna().sum() > 0 ]
sum( dr.isna().sum() > 0 )

# now drop the first fights. 52 still have NAs but the counts did go down.
# based on this I suggest we exclude first-fights from our data.
# these can be modeled separately if we have time.
# takes from ## to ## rows
dr = dr[ dr.R_fighter.duplicated() ]
dr.isna().sum()[ dr.isna().sum() > 0 ]

# drop first fights. this reduces data from over ## to ## rows.
d = d[ ~d.isnull().any(axis=1) | ( d.R_fighter.duplicated() & d.B_fighter.duplicated() ) ].reset_index( drop = True )
d.isna().sum()[ d.isna().sum() > 0 ]

# this leaves only 6 rows with NA referree. Let
# Drop these rows, going from 2356 rows to 2350.
d.dropna( subset = [ 'Referee' ], inplace = True )

# no NA is left:
d.isna().sum()[ d.isna().sum() > 0 ]

# add fight id.
d['fightid'] = range(d.shape[0])

d.to_csv( '../out/b-handle-nas.csv', index = False )

# clean workspace to prep for the next file to run.
del d, dr, f, fmets, fmetsbr, fna, nonafighters