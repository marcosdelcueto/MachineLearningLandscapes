#!/bin/bash
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
# Script to consolidate several logs and save_logs
### START CUSTOMIZABLE DATA ###
log_names='log_grid_l'
save_log_names='save_log_grid_l'
save2_log_names='save2_log_grid_l'
#### END CUSTOMIZABLE DATA ####
# move all logs and save_logs to dummy files
c=0
for i in `ls ${log_names}*`
do
	#echo "Move ${i} to dummy_${c}"
	mv ${i} dummy_${c}
	c=$((c+1))
done
for i in `ls ${save_log_names}*`
do
	#echo "Move ${i} to dummy_${c}"
	mv ${i} dummy_${c}
	c=$((c+1))
done
for i in `ls ${save2_log_names}*`
do
	#echo "Move ${i} to dummy_${c}"
	mv ${i} dummy_${c}
	c=$((c+1))
done
# move dummy files to new logs
c=0
for i in `ls dummy_*`
do
	#echo "Move ${i} to ${log_names}_${c}.log"
	mv ${i} ${log_names}_${c}.log
	c=$((c+1))
done
