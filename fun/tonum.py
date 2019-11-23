# author: Bryce C
# flexibly convert strings to numbers.
 
def tonum( x, nastrings = [ 'nan', '', 'NA', 'NULL' ], dropchars = False ):
    
    ox = x
    x = pd.Series(x)
    if x.dtype != 'O': return x
    
    x[ x.isin( nastrings ) ] = np.nan
    x = x.str.strip()
    x = x.str.replace( '[$,]', '' )
    
    ispct = grep( '%', x )
    x = x.str.replace( '%', '' )
    
    makeneg = grep( '[(].+[)]', x )
    x = x.str.replace( '[()]', '' )
    
    if dropchars: x = x.str.replace( '[^0-9]', '' )
    
    try: 
        x = pd.to_numeric(x)
    except:
        return ox
    
    x.values[ ispct ] = x.values[ ispct ] / 100
    x.values[ makeneg ] = -1 * x.values[ makeneg]
    
    return x

t = [ '$1', '1.00', '25%', '(3.4)', '-1' ]
assert ( tonum(t) == [ 1, 1, .25, -3.4, -1 ] ).all()
t = None

del t