# ./run_fix_baselines.sh

for SETTING in cholec chestx mass_maps supernova politeness emotion 
do
    python fix_baselines.py $SETTING
done