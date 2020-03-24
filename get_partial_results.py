#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
################################################################################
import re
import ast
import statistics

#######################################
### START CUSTOMIZABLE INPUT VALUES ###
Nspf = 100                                      # Number of different landscapes generate
Nwalkers = 2                                    # Number of walkers per landscape
adven = [10,100]                                # percentage of special points per walker
t1_analysis    = True                           # Whether t1 analysis is performed 
t2_exploration = True                           # Whether t2 exploration is performed
log_name = 'log_grid_l'                         # Name of log file. Suffix '_XX.log' is added automatically
pattern = re.compile(r'I am returning these values')
#### END CUSTOMIZABLE INPUT VALUES ####
#######################################
# Initialize values
results_t1_per_Nspf=[]
results_t2_per_Nspf=[]
counter_finished=0
# For each Nspf, check if they were finished, and take results from .log file
for i in range(Nspf):
    filename=log_name + '_' + str(i) + '.log'
    with open(filename) as f:
        for line in f:
            if pattern.search(line): # if final results for the SPF are printed
                # read results and transform into a list
                parsed_line=line.strip().split(':')[1].strip()
                list_line=ast.literal_eval(parsed_line)
                list_line=list(list_line)
                counter_finished=counter_finished+1
                # assign provisional results for each finished SPF 
                if t1_analysis == True:    
                    provi_result_t1=list_line[0]
                    results_t1_per_Nspf.append(provi_result_t1)
                if t2_exploration == True: 
                    provi_result_t2=list_line[1]
                    results_t2_per_Nspf.append(provi_result_t2)
                #print('-----------')
                #print(list_line)
                #print(type(list_line))
                #print(list_line[0])
                #print(list_line[1])
# transpose to get final results per walker
if t1_analysis    == True: results_per_walker_t1=[list(i) for i in zip(*results_t1_per_Nspf)]
if t2_exploration == True: results_per_walker_t2=[list(i) for i in zip(*results_t2_per_Nspf)]
# print final results per SPF and per walker
#print('-----------')
#print('Results per Nspf:')
#print(results_t1_per_Nspf)
#print(results_t2_per_Nspf)
#print('-----------')
#print('Results per walker:')
#print(results_per_walker_t1)
#print(results_per_walker_t2)
print('Number of SPFs checked:', Nspf)
print('Number of SPFs finished:', counter_finished)
# Print final results
print('--- Final results ---',flush=True)
for i in range(Nwalkers):
    print('-- Adventurousness: %6.1f --' %(adven[i]),flush=True)
    if t1_analysis == True:
        print('-- t1 analysis')
        print('- RMSE:',results_per_walker_t1[i][:],flush=True)
        print('- RMSE Mean: %f' %(statistics.mean(results_per_walker_t1[i])),flush=True)
        print('- RMSE Median: %f' %(statistics.median(results_per_walker_t1[i])),flush=True)
    if t2_exploration == True:
        print('-- t2 exploration')
        print('- [ML_gain_pred, ML_gain_real, error_rel_ML]: %s' %(str(results_per_walker_t2[i])),flush=True)       
        ML_gain_pred = [item[0] for item in results_per_walker_t2[i]]
        ML_gain_real = [item[1] for item in results_per_walker_t2[i]] 
        error_rel_ML = [item[2] for item in results_per_walker_t2[i]]
        print('- ML_gain_pred Mean: %f' %(statistics.mean(ML_gain_pred)),flush=True)
        print('- ML_gain_pred Median: %f' %(statistics.median(ML_gain_pred)),flush=True)
        print('- ML_gain_real Mean: %f' %(statistics.mean(ML_gain_real)),flush=True)
        print('- ML_gain_real Median: %f' %(statistics.median(ML_gain_real)),flush=True)
        print('- error_rel_ML Mean: %f' %(statistics.mean(error_rel_ML)),flush=True)
        print('- error_rel_ML Median: %f' %(statistics.median(error_rel_ML)),flush=True)
    print('')
