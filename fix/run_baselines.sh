# ./run_baselines.sh

# for SETTING in cholec chestx mass_maps politeness emotion supernova
for SETTING in chestx
do
    python baselines.py $SETTING
done