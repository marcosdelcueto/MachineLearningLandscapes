#!/bin/bash
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
# Script to save logs that have finished properly
#################################################################################
counter=0
for i in `ls save_log_grid_l_*`
do 
	last=`tail -n 1 ${i} | awk '{print $1}'`
	if [[ ${last} == "I" ]]
	then
		mv ${i} save2_log_grid_l_${counter}.log
		counter=$((${counter}+1))
	fi
done

counter=0
for i in `ls log_grid_l_*`
do 
	last=`tail -n 1 ${i} | awk '{print $1}'`
	if [[ ${last} == "I" ]]
	then
		mv ${i} save_log_grid_l_${counter}.log
		counter=$((${counter}+1))
	fi
done
