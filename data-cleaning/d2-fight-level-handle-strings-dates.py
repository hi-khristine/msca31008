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

d = pd.read_csv( "../out/d1-fight-level-transform.csv" )

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

dd = ddict( d[ d.columns[ d.dtypes == 'object' ] ] )
dd[[ 'name', 'nvals', 'topvals', 'pareto80' ]]

# Take top 10 for columns that have too many values.
for col in [ 'Referee', 'location' ]:
    top10 = d[col].value_counts( ascending = False ).index[0:9]
    d[col].loc[ ~np.isin( d[col], top10 ) ] = 'Other'
    del col, top10
    
# confirm counts.
dd = ddict( d[ d.columns[ d.dtypes == 'object' ] ] )
dd[[ 'name', 'nvals', 'topvals', 'pareto80' ]]

# now get dummies.
d = pd.get_dummies(d)

d.to_csv( '../out/d2-fight-level-handle-strings-dates.csv' )

del d, dd