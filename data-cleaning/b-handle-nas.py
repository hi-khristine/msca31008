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

# check NA count again. it filled in some but lots of NA still.
f[fmets].isna().sum()

# explore NA data.
fna = f[ np.isin( f.fighter, f.fighter[ f.age.isna() ] ) ][ [ 'fighter', 'date' ] + fmets ]

# 35% of fighters have missing physical measurements.
# this is a large percentage but these will be difficult to address so let's drop them for now.
# this reduces data from 5144 to 4073 rows.
nonafighters = f.dropna().fighter
( f.fighter.nunique() - nonafighters.nunique() ) / f.fighter.nunique()
d = d[ np.isin( d.R_fighter, nonafighters ) & np.isin( d.B_fighter, nonafighters ) ].reset_index( drop = True )

# for the fighters left, fill in the data.
# attempt fill.
for i in fmets:
    for br in [ 'B_', 'R_' ]:
        d[ br + i ] = d.groupby( br + 'fighter' )[ br + i ].transform(
            lambda s: np.nan if pd.isnull(s).all() == True else s.loc[ s.first_valid_index() ]
        )
        del br
    del i

# I am guessing NAs come from fights being first fights where no history exists.
d.sort_values( by = [ 'date' ], inplace = True )

# let's look at red: 54 columns have NAs.
dr = d[ grep( '^R_', d.columns, value = True ) ]
dr.isna().sum()[ dr.isna().sum() > 0 ]

# now drop the first fights. 52 still have NAs but the counts did go down.
# based on this I suggest we exclude first-fights from our data.
# these can be modeled separately if we have time.
dr = dr[ dr.R_fighter.duplicated() ]
dr.isna().sum()[ dr.isna().sum() > 0 ]

# drop first fights. this reduces data from over 4073 to 2356 rows.
d = d[ d.R_fighter.duplicated() & d.B_fighter.duplicated() ].reset_index( drop = True )
d.isna().sum()[ d.isna().sum() > 0 ]

# this leaves only 6 rows with NA referree. Let
# Drop these rows, going from 2356 rows to 2350.
d.dropna( subset = [ 'Referee' ], inplace = True )

# no NA is left:
d.isna().sum()[ d.isna().sum() > 0 ]

d.to_csv( '../out/b-handle-nas.csv', index = False )

# clean workspace to prep for the next file to run.
del d, dr, f, fmets, fmetsbr, fna, nonafighters