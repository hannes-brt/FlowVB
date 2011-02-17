import sys
from StringIO import StringIO
import struct
import os

import numpy as np

def load_array_from_fcs( fcs_file_name ):
    """
    Attempts to parse an FCS (flow cytometry standard) file

    Parameters: fcs_file_name
        fcs_file_name: path to the FCS file

    Returns: events
    	events: an [N x D] matrix of the data (as a numpy array)
    	
    	i.e. events[99][2] would be the value at the 3rd dimension
    	of the 100th event
    """
    fcs_file_name = fcs_file_name

    fcs = open( fcs_file_name, 'rb' )
    header = fcs.read( 58 )
    version = header[0:6].strip()
    text_start = int( header[10:18].strip() )
    text_end = int( header[18:26].strip() )
    data_start = int( header[26:34].strip() )
    data_end = int( header[34:42].strip() )
    analysis_start = int( header[42:50].strip() )
    analysis_end = int( header[50:58].strip() )

    print "Parsing TEXT segment"
    # read TEXT portion
    fcs.seek( text_start )
    delimeter = fcs.read( 1 )
    
    # First byte of the text portion defines the delimeter
    print "delimeter:", delimeter
    text = fcs.read( text_end - text_start + 1 )

    #Variables in TEXT poriton are stored "key/value/key/value/key/value"
    keyvalarray = text.split( delimeter )
    fcs_vars = {}
    fcs_var_list = []
    
    # Iterate over every 2 consecutive elements of the array
    for k, v in zip( keyvalarray[::2], keyvalarray[1::2] ):
        fcs_vars[k] = v
        fcs_var_list.append( ( k, v ) ) # Keep a list around so we can print them in order

    #from pprint import pprint; pprint(fcs_var_list)
    if data_start == 0 and data_end == 0:
        data_start = int( fcs_vars['$DATASTART'] )
        data_end = int( fcs_vars['$DATAEND'] )

    num_dims = int( fcs_vars['$PAR'] )
    print "Number of dimensions:", num_dims

    num_events = int( fcs_vars['$TOT'] )
    print "Number of events:", num_events

    # Read DATA portion
    fcs.seek( data_start )
    
    #print "# of Data bytes",data_end-data_start+1
    data = fcs.read( data_end - data_start + 1 )

    # Determine data format
    datatype = fcs_vars['$DATATYPE']
    if datatype == 'F':
        datatype = 'f' # set proper data mode for struct module
        print "Data stored as single-precision (32-bit) floating point numbers"
    elif datatype == 'D':
        datatype = 'd' # set proper data mode for struct module
        print "Data stored as double-precision (64-bit) floating point numbers"
    else:
        assert False, "Error: Unrecognized $DATATYPE '%s'" % datatype
    
    # Determine endianess
    endian = fcs_vars['$BYTEORD']
    
    if endian == "4,3,2,1":
        endian = ">" # set proper data mode for struct module
        print "Big endian data format"
    elif endian == "1,2,3,4":
        print "Little endian data format"
        endian = "<" # set proper data mode for struct module
    else:
        assert False, "Error: This script can only read data encoded with $BYTEORD = 1,2,3,4 or 4,3,2,1"

    # Put data in StringIO so we can read bytes like a file    
    data = StringIO( data )

    print "Parsing DATA segment"
    
    # Create format string based on endianeness and the specified data type
    format = endian + str( num_dims ) + datatype
    datasize = struct.calcsize( format )
    
    print "Data format:", format
    print "Data size:", datasize
    events = []
    
    # Read and unpack all the events from the data
    for e in range( num_events ):
        event = struct.unpack( format, data.read( datasize ) )
        events.append( event )
    
    fcs.close()
    
    events = np.array( events, dtype=np.float64 )
    
    return events
