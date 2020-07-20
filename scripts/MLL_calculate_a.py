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
######   START CUSTOMIZABLE PARAMETERS ########
data_file_name = 'data_x_G'      # prefix of input file name
Nspf = 100                       # number of landscapes
adven = [10,20,30,40,50,60,70,80,90,100]           # array with adventurousness values
#threshold = 0.10
early_stop_parameter = 0.025
intervals = 20
start_t1 = 20
Delta_t1 = 20
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
#array_a = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
#array_a = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
end_t1 = start_t1 + Delta_t1
final_predictions_a=[]
final_median=[]
final_mean=[]
final_stdev=[]
array_a = []
error_mean = []
error_median = []
for i in range(1,intervals+1):
    array_a.append(0.0+i*1/intervals)
print('TEST:', array_a)

for a in adven:
    predictions_a = []
    predictions_a_new = []
    #a = adven[0]
    for l in range(Nspf):
        X  = []
        G  = []
        maximum = []
        adventurousness = []
        f_data = open('%s_%i_%s.dat' % (data_file_name,a,l), 'r')
        for line in f_data:
            provi_X = []
            line = ast.literal_eval(line)
            for i in range(len(line)-1):
                provi_X.append(line[i])
            provi_G = line[i+1]
            X.append(provi_X)
            G.append(provi_G)
        #print(X[10])
        #print(G)
        #print('#################')
        min_dist_per_i=[]
        for i in range(len(G)):
            if i >= start_t1 and i<= end_t1:
                min_dist_each_i=[]
                sorted_X = []
                sorted_G = []
                for j in np.argsort(G[:i]):
                    sorted_X.append(X[j])
                    sorted_G.append(G[j])
                #print('##### Time step number:', i)     # debug
                #print('# Sorted previous points:', i)   # debug
                #for j in range(i):                      # debug
                    #print(sorted_X[j], sorted_G[j])     # debug
                for adventur in array_a:
                    distances = []
                    #print('# Top %f points: %i' %(adventur, int(adventur*i))) # debug
                    #print('Current point:', X[i], G[i]) # debug
                    for j in range(int(adventur*i)):
                        #print('Previous point:', j, sorted_X[j], sorted_G[j]) # debug
                        dist = 0
                        for k in range(len(X[0])):
                            dist = dist + (X[i][k]-sorted_X[j][k])**2
                        dist = math.sqrt(dist)
                        distances.append(dist)
                    min_dist = min(distances)
                    #print('All distances:', distances) # debug
                    #print(100*adventur, '. Minimum distance:', min_dist) # debug
                    min_dist_each_i.append(min_dist)
                #print('min_dist_each_i:',min_dist_each_i) # debug
                min_dist_per_i.append(min_dist_each_i)
        #print('min_dist_per_i:',min_dist_per_i) # debug
        minimum_dist_per_a = [list(i) for i in zip(*min_dist_per_i)]
        #print('minimum_dist_per_a:',minimum_dist_per_a) # debug
        #print('### Landscape %i ###' %(l)) # debug
        next_gradient = []
        for i in range(len(minimum_dist_per_a)):
            maximum.append(max(minimum_dist_per_a[i]))
            adventurousness.append(array_a[i])
            if i != (len(minimum_dist_per_a)-1):
                next_gradient.append(max(minimum_dist_per_a[i]) - max(minimum_dist_per_a[i+1]))
            else:
                next_gradient.append(0.0)
            #print('%.2f %.8f %.4f' %(array_a[i], max(minimum_dist_per_a[i]),next_gradient[i])) # debug
        chosen_a = adventurousness[0]
        chosen_G = maximum[0]
        chosen_a_new = adventurousness[0] + 1/(intervals*2)
        chosen_G_new = next_gradient[0]
        for i in range(len(minimum_dist_per_a)):
            #print('test i', i)
            if maximum[i] < chosen_G:
                chosen_G = maximum[i]
                chosen_a = adventurousness[i] + 1/(intervals*2)
            #for j in range(i):
                #if next_gradient[i] > threshold*next_gradient[j]: 
                    #chosen_G_new = next_gradient[i]
                    #chosen_a_new = adventurousness[i] + 1/(intervals*2)
                    #break
            #if next_gradient[i] > threshold*chosen_G_new:
            if next_gradient[i] > early_stop_parameter:
                chosen_G_new = next_gradient[i]
                chosen_a_new = adventurousness[i] + 1/(intervals*2)
        predictions_a.append(chosen_a)
        predictions_a_new.append(chosen_a_new)
        #print('Chosen "identical" values %.3f %.8f' %(chosen_a, chosen_G))        # debug
        #print('Chosen "gradient" values %.3f %.8f' %(chosen_a_new, chosen_G_new)) # debug
    #print('Final predictions:', predictions_a)
    #print('Final predictions new:', predictions_a_new)
    if Nspf > 1:
        print('######## a =', a, '########')
        print('"Identical values:"')
        print('Mean:', 100*statistics.mean(predictions_a))
        print('Median:', 100*statistics.median(predictions_a))
        print('Stdev',100*statistics.stdev(predictions_a))
        print('"Threshold values:"')
        print('Mean:', 100*statistics.mean(predictions_a_new))
        print('Median:', 100*statistics.median(predictions_a_new))
        print('Stdev',100*statistics.stdev(predictions_a_new))
        print('#######################################3')
        final_predictions_a.append(predictions_a_new)
        final_median.append(100*statistics.median(predictions_a_new))
        final_mean.append(100*statistics.mean(predictions_a_new))
        final_stdev.append(100*statistics.stdev(predictions_a_new))
    print('Error mean:', abs(a-100*statistics.mean(predictions_a_new)))
    print('Error median:', abs(a-100*statistics.median(predictions_a_new)))
    error_mean.append(abs(a-100*statistics.mean(predictions_a_new)))
    error_median.append(abs(a-100*statistics.median(predictions_a_new)))

# Box plot
pntbox=plt.boxplot(final_predictions_a,patch_artist=True,labels=adven,showfliers=False)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Real $a$ (%)',fontsize=15)
plt.ylabel('Estimated $a$ (%)',fontsize=15)
file1='estimation_a.png'
plt.savefig(file1,format='png',dpi=600)
print('save box plot to %s' %file1,flush=True)
plt.close()

# Error bars
x = adven
y = final_median
#y = final_mean
e = final_stdev
print('x:')
print(x)
print('y:')
print(y)
print('e:')
print(e)
plt.errorbar(x, y, e, linestyle='None', marker='o',solid_capstyle='projecting', capsize=5)
plt.plot(adven,adven, 'r-') 
plt.xticks(np.arange(10,110,10),fontsize=10)
plt.yticks(np.arange(10,110,10),fontsize=10)
plt.xlabel('Real $a$ (%)',fontsize=15)
plt.ylabel('Estimated $a$ (%)',fontsize=15)
plt.ylim(0,110)
file2='estimation_a_2.png'
plt.savefig(file2,format='png',dpi=600)
print('save plot to %s' %file2,flush=True)
plt.close()
print('Average Error mean:')
print(sum(error_mean)/len(error_mean))
print('Average Error median:')
print(sum(error_median)/len(error_median))
