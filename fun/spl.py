from random import sample, choices

def spl( x, n = 10, warn = True, replace = False ):

    if n < 1: n = math.ceil( n * nrow(x) )

    if not replace:
      
      if n > nrow(x) and warn: print(
        'easyr::spl: You have sampled more [%s] than the number of available items [%s] and chosen replace = FALSE. Returning the maximum available rows instead.' % ( n, nrow(x) )
      )

      n = min( [ n, nrow(x) ] )
      
      return x.iloc[ sample( range(nrow(x)-1), n ) , : ].reset_index( drop = True )

    else:
        return x.iloc[ choices( range( nrow(x)-1 ), n ), : ].reset_index( drop = True )
    
