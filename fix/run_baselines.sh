# ./run_baselines.sh

for SETTING in cholec chestx mass_maps supernova politeness emotion 
do
    python baselines.py $SETTING
done