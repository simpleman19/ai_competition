#!/usr/bin/env bash
zone=us-west1-b
name=instance-1
file="train_list"
host=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)
uuid=$2
mf=$3

if [ -z "$host" ]; then
echo "Starting Instance"
gcloud -q compute instances start ${name} --zone=${zone}
sleep 60
host=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)
fi

scp -i ~/.ssh/google_compute_engine "archive/${mf}" chancert413_gmail_com@${host}:'/home/chancert413_gmail_com/ai_competition/'

ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host} bash -c "'
cd /home/chancert413_gmail_com/ai_competition
pwd
. .env/bin/activate
git checkout master
git pull
git checkout ${uuid}
pip install -r requirements.txt
python3.6 train.py ${uuid} \"${mf}\"'"

scp -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host}:'/home/chancert413_gmail_com/ai_competition/*.{h5,png,log}' archive/
ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host} bash -c "'
rm /home/chancert413_gmail_com/ai_competition/*.png
rm /home/chancert413_gmail_com/ai_competition/*.log
rm /home/chancert413_gmail_com/ai_competition/*.temp
rm /home/chancert413_gmail_com/ai_competition/*.npy
'"
rm train_list
touch train_list

if [ "$1" == "-d" ]; then
    if [ -z "$host" ]; then
        echo "Didn't find instance to delete"
    else
        echo "Stopping Instance: "
        ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host} bash -c "'
sudo shutdown -h now
'"
        ssh-keygen -R ${host}
    fi
fi