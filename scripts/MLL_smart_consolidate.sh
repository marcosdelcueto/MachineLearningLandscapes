#!/bin/bash

counter_adven4=0
counter_adven10=0
counter_wtf=0

for i in `ls log*`
do
	last_line=`tail -n 1 ${i} | cut -d "[" -f 2 | awk -F\, '{print NF-1}'`
	#echo ""
	#echo ${last_line}
	if [[ ${last_line} == 4 ]] 
	then
		#echo "This is 4, ${counter_adven4}"
		mv ${i} save4_log_grid_l_${counter_adven4}.log
		counter_adven4=$((${counter_adven4}+1))
	elif [[ ${last_line} == 10 ]]
	then
		#echo "This is 10, ${counter_adven10}"
		mv ${i} save10_log_grid_l_${counter_adven10}.log
		counter_adven10=$((${counter_adven10}+1))
	else
		echo "WTF is going on"
		mv ${i} save_wtf_og_grid_l_${counter_wtf}.log 
		counter_wtf=$((${counter_wtf}+1))
	fi
done

counter=0
for i in `ls save10*`
do
	mv ${i} log_grid_l_${counter}.log
	counter=$((${counter}+1))
done
#counter=100
for i in `ls save4*`
do
	mv ${i} log_grid_l_${counter}.log
	counter=$((${counter}+1))
done
