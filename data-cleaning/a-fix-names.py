# author: Bryce C
# purpose: check text values in the raw data for issues.
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

d = pd.read_csv('../data/data.csv')

# add fight id early on.
d['fightid'] = range(d.shape[0])
d.date = pd.to_datetime(d.date)

# add odds.
if ( 'iterationnum' not in globals() ) or (iterationnum >= 2 ):
    load( '../scrape-odds/get-fight-odds.pkl' )
    d = pd. merge( d, f[[ 'fightid', 'R_odds', 'B_odds' ]], how = 'left', on = 'fightid' )
    d.R_odds = tonum(d.R_odds)
    d.B_odds = tonum(d.B_odds)

# fill missing referrees.
# 23 missing referees.
d.Referee.isna().sum()
mr = pd.read_csv( '../data/Missing_Referees_Scraped.csv' )
mr.date = pd.to_datetime( mr.date )
for index, row in mr.iterrows():
    d.loc[
        ( d.R_fighter == row.R_fighter ) & ( d.B_fighter == row.B_fighter ) & ( d.date == row.date),
        d.columns == 'Referee'
    ] = row.Referee
    del index, row
# 1 missing referee.
d.Referee.isna().sum()
del mr

# I extract the text values and check them for typos using OpenRefine faceting.
doextracttext = False
if doextracttext:
    
    text = d.B_fighter.values
    for i in [ 'R_fighter', 'Referee', 'location', 'weight_class' ]:
        text = np.append( text, d[i].values )
        del i
    
    text = pd.Series( text ).unique()
    pd.DataFrame({ 'text': text }).to_csv( 'text.csv', index = False )
    del text
    
del doextracttext
    
# I checked them and only found a few corrections, which I saved here:

repl = [[ 'Berlin, Berlin, Germany',	'Berlin, Germany' ]]
for item in repl: d.location.replace( item[0], item[1] )

# confirm replacement success.
for item in repl: 
    if ( d.location.str.find(item[0]) > 0 ).any(): raise Exception( "Replacement Failed." )

del item, repl
        
# check other columns that have fewer values.
strings = d.dtypes.index[ d.dtypes == 'object' ] 
strings = setdiff( strings, [ 'B_fighter', 'R_fighter', 'Referee', 'location', 'weight_class' ] )

# date we can convert to a date later.
d.Winner.value_counts()
d.B_Stance.value_counts()
d.R_Stance.value_counts()

# these look good.

# save the data. 
d.to_csv( '../out/a-fix-names.csv', index = False )

# clean workspace to prep for the next file to run.
del d, strings