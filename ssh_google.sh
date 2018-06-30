#!/usr/bin/env bash
zone=us-west1-b
name=instance-1
file="train_list"
host=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)

ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host}