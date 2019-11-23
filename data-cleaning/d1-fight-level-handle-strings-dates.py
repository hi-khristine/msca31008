# author: Bryce C
# purpose: handle categorical and date columns in fight-level data.
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

d = pd.read_csv( '../out/b-handle-nas.csv' )

# don't need fighter any more.
d.drop( [ 'R_fighter', 'B_fighter' ], axis = 1, inplace = True )

# convert date to parts.
# these are months, without date so just get month and year.
d.date = pd.to_datetime( d.date )
for col in d.columns[ d.dtypes == 'datetime64[ns]' ]:
    d[ col + '_year' ] = d[col].dt.year
    d[ col + '_month' ] = d[col].dt.month
    d[ col + '_dayofmonth' ] = d[col].dt.day
    d[ col + '_dayofweek' ] = d[col].dt.dayofweek
    d[ col + '_frisat' ] = np.isin( d[ col + '_dayofweek' ], [4,5] )
    d.drop( col, axis = 1, inplace = True )
    del col

# we need to convert stance to an R/B-indepenent value.
# let's take the values, sort them alphabetically, and concatenate them.
d['Stance'] = '(Missing)'
stancecol = which( d.columns == 'Stance' )
for index, row in d.iterrows():
    stances = [ d.R_Stance[index], d.B_Stance[index] ]
    stances.sort()
    d.iloc[ index, stancecol ] = '-'.join( stances )
    del index, stances, row
del stancecol
d.drop( [ 'R_Stance', 'B_Stance' ], axis = 1, inplace = True )

# identify women's fights and drop this from categorization.
d['womens'] = d.weight_class.str.contains('Women\'s')
d.weight_class = d.weight_class.str.replace('Women\'s ', '')

# change weight class into number since it has a direction.
# Catch/Open Weight isn't ordered and there are only a few records that have it.
# https://en.wikipedia.org/wiki/Mixed_martial_arts_weight_classes
d.weight_class.value_counts()
d['weight_class_catch_weight' ] = d.weight_class == "Catch Weight"
d['weight_class_open_weight' ] = d.weight_class == "Open Weight"
ordered_classes = [ "Catch Weight", "Open Weight", 'Strawweight', 'Flyweight', 'Bantamweight',  'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight' ]
d.weight_class = [
        ordered_classes.index(x)
        for x in d.weight_class
]

# we want to differentiate open and catch weight so set them to -1.
# We have other columns that will differentiate them (weight_class_catch_weight, weight_class_open_weight)
d.loc[ np.isin( d.weight_class, [0, 1] ), 'weight_class' ] = -1
del ordered_classes 

# convert Winner to 0/1. Blue or Draw will be 0, meaning Red wins = 1.
d.Winner = [ 1 if x == 'Red' else 0 for x in d.Winner ]

dd = ddict( d[ d.columns[ d.dtypes == 'object' ] ] )
dd.sort_values( 'nvals', axis = 0, ascending = False )[[ 'name', 'nvals', 'topvals', 'pareto80' ]]

# Take top 30 for columns that have too many values.
for col in [ 'Referee', 'location' ]:
    top30 = d[col].value_counts( ascending = False ).index[0:29]
    d.loc[ ~np.isin( d[col], top30 ), col ] = 'Other'
    del col, top30
    
# confirm counts. these will be changed to dummies:
dd = ddict( d[ d.columns[ d.dtypes == 'object' ] ] )
dd[[ 'name', 'nvals', 'topvals', 'pareto80' ]]

# now get dummies.
d = pd.get_dummies(d)

d.to_csv( '../out/d1-fight-level-handle-strings-dates.csv', index = False )

del d, dd