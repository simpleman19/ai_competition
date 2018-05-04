#!/usr/bin/env bash
uuid=$(uuidgen)
git tag ${uuid}
git push origin --tags
ssh 198.204.229.156 bash -c "'
cd /home/simpleman19/ai_competition
pwd
. .env/bin/activate
pip install -r requirements.txt
git checkout master
git pull
git checkout ${uuid}
python3.6 train.py ${uuid}'"