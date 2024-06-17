# ./run_fix_baselines.sh

# for SETTING in cholec chestx #mass_maps supernova multilingual_politeness emotion 
for SETTING in mass_maps
do
    python fix_baselines.py $SETTING
done