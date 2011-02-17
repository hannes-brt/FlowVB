'''
Created on 2011-02-16

@author: Andrew Roth
'''
import csv

import numpy as np

from flowvb.fcs import load_array_from_fcs
from flowvb.core.flow_vb import FlowVBAnalysis

def run_flow_vb( args ):
    if args.file_format == 'csv':
        data = load_array_from_csv( args.inpute_file_name )
    elif args.file_format == 'fcs':
        data = load_array_from_fcs( args.inpute_file_name )
    
    analysis = FlowVBAnalysis( data )

def write_outpus( args, analysis ):    
    soft_labels = analysis.get_soft_labels()
    soft_labels_file_name = args.prefix + ".soft_labels.tsv"  
    write_array_to_csv( soft_labels_file_name, soft_labels )
    
    labels = analysis.get_labels()
    labels_file_name = args.prefix + ".labels.tsv"
    write_array_to_csv( labels_file_name, labels )
    
def write_array_to_csv( file_name, soft_labels ):
    writer = csv.writer( open( file_name, 'w' ), delimiter='\t' )
    writer.writerows( soft_labels )
    


def load_array_from_csv( csv_file_name ):
    reader = csv.reader( open( csv_file_name ), delimiter=' ' )
    
    rows = []
    
    for row in reader:
        rows.append( [float( x ) for x in row] )
    
    data = np.array( rows, dtype=np.float64 )
    
    return data

if __name__ == "__main__":
    faithful_csv = '../tests/data/old_faithful/faithful.txt'
    
    print load_array_from_csv( faithful_csv )
