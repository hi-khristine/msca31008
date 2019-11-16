# author: Bryce C working off code from Khristine.
# purpose: add dummy variables to replace categorical.
# note: 
#   this code should be run from the data-cleaning directory.
#   when it runs, you'll have a corrected CSV file in the out/ folder
#   called c3-figher-level-dummies.csv

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

load( '../out/c2-fighter-level-fillna.pkl' )

fighter_dataset=pd.get_dummies(fighter_dataset, columns=['Stance'])

save( '../out/c3-fighter-level-dummies.pkl', fighter_dataset, na_num_cols  )