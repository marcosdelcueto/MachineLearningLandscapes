#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
# This script plots boxplots of rmse read from stdout (obtained either directly from program or mimicked with 'get_log_results.py' from logs)
################################################################################
import re
import ast
import sys
import statistics
import matplotlib.pyplot as plt

#######################################
### START CUSTOMIZABLE INPUT VALUES ###
Nwalkers = 10                                        # Number of walkers per landscape
adven = [10,20,30,40,50,60,70,80,90,100]             # percentage of special points per walker
#out_file = 'slurm-27819080.out'                       # Name of stdout file
out_file = str(sys.argv[1])                       # Name of stdout file
print('Out file:', out_file)
pattern = re.compile(r'RMSE:') # Flag used to identify results in .log file
#### END CUSTOMIZABLE INPUT VALUES ####
#######################################
def main(Nwalkers,adven,out_file,pattern):
    counter = 0
    results_per_walker_t1 = []
    with open(out_file) as f:
        for line in f:
            if pattern.search(line): # if we are in line with RMSE
                rmse = []
                line = re.findall(r'\[(.*?)\]', line)
                line=line[0].split(',')
                for item in line:
                    rmse.append(float(item))
                #print('-- a = ',adven[counter])
                #print(rmse)
                #print(len(rmse))
                #plot_rmse('rmse',rmse)
                counter=counter+1
                results_per_walker_t1.append(rmse)
        plot_rmse('rmse',results_per_walker_t1)

def plot_rmse(flag,results_per_walker_t1):
    if flag=='rmse':
        pntbox=plt.boxplot(results_per_walker_t1,patch_artist=True,labels=adven,showfliers=False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (a.u.)',fontsize=15)
        file1='rmse.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1,flush=True)
        plt.close()

        pntbox=plt.boxplot(results_per_walker_t1,patch_artist=True,labels=adven,showfliers=True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (a.u.)',fontsize=15)
        file1='rmse_with_outliers.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1,flush=True)
        plt.close()



########################
main(Nwalkers,adven,out_file,pattern)
