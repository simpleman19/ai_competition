#!/usr/bin/env bash
file="train_list"
lines=`cat ${file}`
for uuid in ${lines}; do
        ssh 198.204.229.156 bash -c "'
cd /home/simpleman19/ai_competition
pwd
. .env/bin/activate
git checkout master
git pull
git checkout ${uuid}
pip install -r requirements.txt
python3.6 train.py ${uuid} > ${uuid}.log'"
done

scp 198.204.229.156:'/home/simpleman19/ai_competition/*.{h5,png,log}' archive/
ssh 198.204.229.156 bash -c "'
rm /home/simpleman19/ai_competition/*.h5
rm /home/simpleman19/ai_competition/*.png
rm /home/simpleman19/ai_competition/*.log
'"
rm train_list
touch train_list