'''
Created on 2011-02-16

@author: Andrew Roth
'''
import csv

import numpy as np

from flowvb.fcs import load_array_from_fcs
from flowvb.core.flow_vb import FlowVBAnalysis
from flowvb.initialize import D2Initialiser, KMeansInitialiser, RandomInitialiser, UserParameterInitialiser

def run_flow_vb(args):
    if args.file_format == 'csv':
        data = load_array_from_csv(args.input_file_name)
    elif args.file_format == 'fcs':
        data = load_array_from_fcs(args.input_file_name)
    
    # If not file with init params supplied look for initialisation method.
    if args.init_params_file is None:   
        if args.init_method == 'd2-weighting':
            initialiser = D2Initialiser()
        elif args.init_method == 'kmeans':
            initialiser = KMeansInitialiser()
        elif args.init_method == 'random':
            initialiser = RandomInitialiser()
            
        args.init_params = initialiser.initialise_parameters(data, args.num_comp_init)
    # Else load initial parameters from file.
    else:
        initialiser = UserParameterInitialiser()
        
        args.init_params = initialiser.initialise_parameters(data, args.init_params_file)
        
    analysis = FlowVBAnalysis(data, args)
    
    write_analysis_to_files(args.output_prefix, analysis)

#=======================================================================================================================
# IO
#=======================================================================================================================
def write_analysis_to_files(prefix, analysis):    
    soft_labels = analysis.get_soft_labels()
    soft_labels_file_name = prefix + ".soft_labels.tsv"  
    write_array_to_csv(soft_labels_file_name, soft_labels)
    
    labels = analysis.get_labels()
    labels_file_name = prefix + ".labels.tsv"
    write_vector_to_file(labels_file_name, labels)
    
def write_array_to_csv(file_name, array):
    writer = csv.writer(open(file_name, 'w'), delimiter='\t')
        
    writer.writerows(array)

def write_vector_to_file(file_name, vector):
    fh = open(file_name, 'w')
    
    for entry in vector:
        fh.write(str(entry) + '\n')
    
    fh.close()

def load_array_from_csv(csv_file_name):
    reader = csv.reader(open(csv_file_name), delimiter=',')
    
    header = reader.next()
    
    rows = []
        
    for row in reader:
        rows.append([float(x) for x in row])
    
    data = np.array(rows, dtype=np.float64)
    
    return data

if __name__ == "__main__":
    faithful_csv = '../tests/data/old_faithful/faithful.txt'
    
    print load_array_from_csv(faithful_csv)
