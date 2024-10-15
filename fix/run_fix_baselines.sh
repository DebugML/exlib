# ./run_fix_baselines.sh

for SETTING in multilingual_politeness #mass_maps #chestx #mass_maps #cholec chestx mass_maps supernova multilingual_politeness emotion
do
    python fix_baselines.py $SETTING
done
