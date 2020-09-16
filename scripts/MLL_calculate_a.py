#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
# Script to estimate adventurousness 'a' from datasets
#################################################################################
import ast
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ######
data_file_name = 'data_x_G'                 # prefix of input file name
Nspf = 100                                  # number of landscapes
adven = [10,20,30,40,50,60,70,80,90,100]    # different a values from datasets
early_stop_parameter = 0.025                # threshold below which max-min-dist variations are ignored
Delta_a = 5                                 # Delta_a value of a-grid
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
intervals = int(100/Delta_a) # number of intervals of our a-grid
start_t1=intervals           # to be able to calculate points in 'Delta_a' percent, we need 'start_t1' points
# Initialize lists
final_predictions_a=[]
final_median=[]
final_stdev=[]
array_a = []
# Create a-grid
for i in range(1,intervals+1):
    array_a.append(0.0+i*1/intervals)
# Loop for each adventurousness values
for a in adven:
    predictions_a = []
    predictions_a_new = []
    # Loop for each dataset
    for l in range(Nspf):
        X  = []
        G  = []
        f_data = open('%s_%i_%s.dat' % (data_file_name,a,l), 'r')
        # Read data
        for line in f_data:
            provi_X = []
            line = ast.literal_eval(line)
            for i in range(len(line)-1):
                provi_X.append(line[i])
            provi_G = line[i+1]
            X.append(provi_X)
            G.append(provi_G)
        min_dist_per_i=[]
        # For each timestep
        for i in range(len(G)):
            if i >= start_t1:
                min_dist_each_i = []
                sorted_X = []
                sorted_G = []
                # Sort previous points from smallest G to largest one
                for j in np.argsort(G[:i]):
                    sorted_X.append(X[j])
                    sorted_G.append(G[j])
                # Loop for each interval in our a-grid
                for adventur in array_a:
                    distances = []
                    # Loop for each of the previous points in top 'adventure' percent
                    for j in range(int(adventur*i)):
                        dist = 0
                        # Calculate distance of point i with previous point j
                        for k in range(len(X[0])):
                            dist = dist + (X[i][k]-sorted_X[j][k])**2
                        dist = math.sqrt(dist)
                        distances.append(dist)
                    # Choose minimum distance between i and any j
                    min_dist = min(distances)
                    # Append to list with min_dist for each timestep (i)
                    min_dist_each_i.append(min_dist)
                # Append to list with sub-list with min_dist for each timestep (i)
                min_dist_per_i.append(min_dist_each_i)
        # Transpose list, so it contains sub-list with min_dist for each interval in a-grid
        minimum_dist_per_a = [list(i) for i in zip(*min_dist_per_i)]
        next_gradient = []
        # Loop for each a-grid point
        for i in range(len(minimum_dist_per_a)):
            # Calculate gradient of max(min_dist) between points of a-grid: 0 if last
            if i != (len(minimum_dist_per_a)-1):
                next_gradient.append(max(minimum_dist_per_a[i]) - max(minimum_dist_per_a[i+1]))
            else:
                next_gradient.append(0.0)
            #print('%.2f %.8f %.4f' %(array_a[i], max(minimum_dist_per_a[i]),next_gradient[i])) # debug
        # Calculate chosen adventure as the point in between two a-grid values where the variation of gradient is below a given threshold
        chosen_a_new = array_a[0] + 1/(intervals*2)
        for i in range(len(minimum_dist_per_a)):
            if next_gradient[i] > early_stop_parameter:
                chosen_a_new = array_a[i] + 1/(intervals*2)
        # Append predicted a value for each SPF
        predictions_a_new.append(chosen_a_new)
        #print('Chosen "gradient" values %.3f' %(chosen_a_new)) # debug
    if Nspf > 1:
        print('a = %4i . Median: %8.4f . Stdev: %8.4f' % (a, statistics.median(predictions_a_new), statistics.stdev(predictions_a_new)))
        # Transform to percentage
        for i in range(len(predictions_a_new)):
            predictions_a_new[i] = predictions_a_new[i] *100
        # Append to final arrays for plots
        final_predictions_a.append(predictions_a_new)
        final_median.append(statistics.median(predictions_a_new))
        final_stdev.append(statistics.stdev(predictions_a_new))
print('########################################')
# Box plot
#pntbox=plt.boxplot(final_predictions_a,patch_artist=True,labels=adven,showfliers=False)
#plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)
#plt.xlabel('Real $a$ (%)',fontsize=15)
#plt.ylabel('Estimated $a$ (%)',fontsize=15)
#file1='estimation_a_boxplot.png'
#plt.savefig(file1,format='png',dpi=600)
#print('save box plot to %s' %file1,flush=True)
#plt.close()

# Errorbar plot
x = adven
y = final_median
e = final_stdev
plt.errorbar(x, y, e, linestyle='None', marker='o',solid_capstyle='projecting', capsize=5)
plt.plot(adven,adven, 'r-') 
plt.xticks(np.arange(10,110,10),fontsize=10)
plt.yticks(np.arange(10,110,10),fontsize=10)
plt.xlabel('Real $a$ (%)',fontsize=15)
plt.ylabel('Estimated $a$ (%)',fontsize=15)
plt.ylim(0,110)
file2='estimation_a_errorbar.png'
plt.savefig(file2,format='png',dpi=600)
print('save errorbar plot to %s' %file2,flush=True)
plt.close()
