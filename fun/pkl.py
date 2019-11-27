from inspect import stack
import re
import pickle

def save( path, *argv ):
    
    # get arg names from the function call.
    names = re.search( '\(([^)]+)\)', stack()[1].code_context[0] ).group(1)
    
    # was the path name used? if so, remove it.
    if re.search( 'path *=', names ): 
        names = names.replace( 'path *=', '' )
        
    # otherwise, path is the first argument. remove it.
    else: 
        names = re.sub( '^[^,]+,', '', names )
    
    # then split by comma and remove blanks.    
    names = [ i for i in [ i.strip() for i in names.split(',') ] if i != '' ]
    
    obj = {}
    for i in range(len(argv)): 
        obj[ names[i] ] = argv[i]
    
    output = open( path, 'wb' )
    pickle.dump( obj, output )
    output.close()
    
def load( path ):
    
    pkl_file = open( path, 'rb' )
    dt = pickle.load(pkl_file)
    pkl_file.close()
    
    for i in dt: globals()[i] = dt[i]