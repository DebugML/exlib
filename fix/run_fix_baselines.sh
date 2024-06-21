# ./run_fix_baselines.sh

for SETTING in mass_maps supernova multilingual_politeness emotion chestx cholec
do
    python fix_baselines.py $SETTING
done
