# author: Bryce C
# purpose: check text values in the raw data for issues.
# note: 

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

d = pd.read_csv('../out/data-namefix.csv')
#dd = ddict(d)

s = spl(d)
s.to_csv('sample.csv')
