# author: Bryce C
# build data dictionary for a DataFrame.

from random import sample
from math import ceil
from numpy import nan, cumsum, issubdtype, number

def ddict( x, nastrings = ['NA', 'NULL', ''], runonsample = 1 ):
#if True: 
    
    #x = X
    #nastrings = ['NA', 'NULL', '']
    #runonsample = 1

    ddict = pd.DataFrame()
    x = x.reset_index( drop = True )
    
    # track original row.
    x['bcrow']= range( x.shape[0] )
    
    if runonsample < 1: x = x.iloc[ sample( range( x.shape[0] ), ceil( x.shape[0] * runonsample ) ), : ].reset_index( drop = True )

    # sample rows first so values are consistent across.
    splrows = sample( range( x.shape[0] ), min( 5, x.shape[0] ) )

    for col in [ i for i in x.columns if i != 'bcrow' ]:

        coldict = pd.DataFrame({
            'name' : [col], 
            'dtype' : [str(x[col].dtype)],
            'sample' : nan,
            'topvals' : nan,
            'nvals' : nan,
            'unique': nan,
            'napct' : nan,
            'min': nan,
            'max' : nan,
            'outlierct' : nan,
            'mean' : nan,
            'mode': nan,
            'pct25' : nan,
            'pct50' : nan,
            'pct75' : nan,
            'pareto80': nan,
            'outlierpct' : nan,
            'outlierrows' : nan,
            'narows' : nan,
            'sd' : nan
        })

        # replace na strings and relevel.
        if x[col].dtype != 'O' and len(nastrings) > 0:
          
            for ina in nastrings: 
                newna = which( x[col] == ina )
                if len(newna) > 0: x[col].iloc[ newna ] = nan                
                del ina, newna
        
        navals = which( x[col].isna() )
        coldict['unique'] = nan if x[col].duplicated().any() else 'unique'

        if len(navals) > 0:
            vals = x.iloc[ pd.Int64Index(range(len(x))).difference(navals), : ][['bcrow', col ]].reset_index(drop = True )
        else:
            vals = x[[ 'bcrow', col ]]
          
        valcnt = x[col].value_counts( ascending = False )
        if len(valcnt) > 0: 
            coldict['mode'] = str( valcnt.index[0] )
            coldict['topvals'] = ', '.join( pd.Series( valcnt.index ).astype(str).head(5) )
            coldict['pareto80'] = round( ( sum( cumsum(valcnt) < ( len(vals) * 0.8) ) + 1 ) / len(valcnt), 2 )
            coldict['nvals'] = len(valcnt)            
        del valcnt
          
        if x[col].dtype == 'bool':
          
          coldict['mean'] = vals[col].mean()
          coldict['sd'] = vals[col].std()
          coldict['nvals'] = vals[col].nunique()

        # we'll assume this is a date or a number.
        else:
          
            vals = vals.sort_values(by=[col])

            if vals.shape[0] > 0: 
                coldict['nvals'] = vals[col].nunique()
                coldict['min'] = vals[col][0]
                coldict['pct25'] = vals[col].iloc[ ceil( vals.shape[0] * .25 ) - 1 ]    
                coldict['pct50'] = vals[col].iloc[ ceil( vals.shape[0] * .5 ) - 1 ]
                coldict['pct75'] = vals[col].iloc[ ceil( vals.shape[0] * .75 ) - 1 ]
                coldict['max'] = vals[col].iloc[ vals.shape[0] - 1 ]
            
            if issubdtype(x[col].dtype, number):
              
              iqr = coldict['pct75'] - coldict['pct25']      
              outlierrows = vals.bcrow.iloc[ which( vals[col].values > ( coldict['pct75'] + iqr )[0] ) ]
              coldict['outlierct'] = len(outlierrows)
              coldict['outlierpct'] = coldict['outlierct'] / vals.shape[0]
              coldict['outlierrows'] = nan
              if len(outlierrows) > 0: coldict['outlierrows'] = ', '.join( [ str(i) for i in outlierrows.head(5).values ] )
              del outlierrows

              coldict['mean'] = vals[col].mean()
              coldict['sd'] = vals[col].std()

        # all types.
        coldict['nact'] = len(navals)
        coldict['napct'] = len(navals) / x.shape[0]
        coldict['narows'] = nan
        if len(navals) > 0: coldict['narows'] = ', '.join( [ str(i) for i in pd.Series(navals).head(5) ] )            
        coldict['unique'] = coldict['nvals'] == x.shape[0] - coldict['nact']
        
        coldict['sample'] = ', '.join( x[col].astype(str).iloc[ splrows ] )
              
        for i in [ i for i in coldict.columns if i not in [ 'unique' ]  ]:
            if issubdtype(coldict[i].dtype, number): 
                coldict[i] = str( coldict[i][0] )
            elif coldict[i].dtype != 'O':
                coldict[i] = coldict[i].to_string()
            del i

        ddict = pd.concat( [ddict, coldict], sort = False )

        del coldict, col, navals

    for i in ddict.columns: ddict[i] = tonum( ddict[i] )
    del i
    
    return(ddict)