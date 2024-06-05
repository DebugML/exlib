# ./run_baselines.sh

# for SETTING in cholec chestx mass_maps supernova politeness emotion 
for SETTING in cholec chestx mass_maps supernova
do
    python baselines.py $SETTING
done