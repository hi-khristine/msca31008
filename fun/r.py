# author: Bryce C
# R functions for python.

from re import search
from itertools import compress
from numpy import isnan

def nrow(x): return x.shape[0]
def ncol(x): return x.shape[1]
def which(x): return list( compress( range(len(x)), x ) )
def stop(x): raise Exception(x)
def grepl( pattern, x, nafalse = True ): 
    
    val = pd.Series(x)
    val = val.str.contains( pattern )
    if nafalse: val[ val.isnull() ] = False
    return val.tolist()

def grep( pattern, x, naskip = True, value = False ): 
    idx = which( grepl( pattern = pattern, x = x, nafalse = naskip ) )
    if value: 
        return x[ idx ]
    else:
        return idx
    
def trimws(x): return strip(x)
def setdiff( x, y ): return [ i for i in x if i not in y ]
def table(x): return x.value_counts( ascending = False )