#!/bin/bash
#SBATCH --job-name=run_array
#SBATCH --output=slurm_outputs/run_array_%j.out
#SBATCH --error=slurm_outputs/run_array_%j.err
#SBATCH --array=0-26
#SBATCH -x node[093,094,105]
#SBATCH -t 02:00:00
#SBATCH --mem 40G
#SBATCH -n 1

i=0
for mod in "bert-large-cased" "roberta-large" ; do #"bert-large-cased" "gpt-j" "gpt2-xl" "roberta-large" ; do
    for dat in "EventsAdapt" "EventsRev" "DTFit" ; do
        if [ $dat == "EventsAdapt" ] ; then
            for vt_st in 'normal+normal' 'active-active+AAN-AAN' 'active-active+AAN-AAR' 'active-active+AAN-AI' \
                         'active-active+AI-AAN' 'active-active+AI-AAR' 'active-active+AI-AI' 'active-active+normal-AAR' \
                         'active-active+normal' 'active-passive+normal' 'passive-active+normal' 'passive-passive+normal' ; do
                model_list[$i]=$mod
                dataset_list[$i]=$dat
                vt_st_list[$i]=$vt_st
                i=$[$i+1]
            done
        else
            for vt_st in 'normal+normal' ; do
                model_list[$i]=$mod
                dataset_list[$i]=$dat
                vt_st_list[$i]=$vt_st
                i=$[$i+1]
            done
        fi
    done
done

timestamp() {
  date +"%T"
}

source /om2/user/ckauf/anaconda/etc/profile.d/conda.sh
conda activate events3.8
echo "Environment ready!"

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "voice type / sentence type ${vt_st_list[$SLURM_ARRAY_TASK_ID]}"

timestamp

filename="/om2/user/ckauf/lm-event-knowledge/probing/scripts/slurm_outputs/${model_list[$SLURM_ARRAY_TASK_ID]}_${dataset_list[$SLURM_ARRAY_TASK_ID]}_${vt_st_list[$SLURM_ARRAY_TASK_ID]}_${SLURM_ARRAY_TASK_ID}.txt"

python regression_model2plausibility.py --model_name ${model_list[$SLURM_ARRAY_TASK_ID]} --dataset_name ${dataset_list[$SLURM_ARRAY_TASK_ID]} --condition ${vt_st_list[$SLURM_ARRAY_TASK_ID]} > $filename

timestamp
echo 'All complete!'
