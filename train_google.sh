#!/usr/bin/env bash
host=$1
file="train_list"
lines=`cat ${file}`
for uuid in ${lines}; do
        ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host} bash -c "'
cd /home/chancert413_google_com/ai_competition
pwd
. .env/bin/activate
git checkout master
git pull
git checkout ${uuid}
pip install -r requirements.txt
python3.6 train.py ${uuid} > ${uuid}.log'"
done

ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host}:'/home/chancert413_gmail_com/ai_competition/*.{h5,png,log}' archive/
ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host} bash -c "'
rm /home/simpleman19/ai_competition/*.h5
rm /home/simpleman19/ai_competition/*.png
rm /home/simpleman19/ai_competition/*.log
'"
rm train_list
touch train_list