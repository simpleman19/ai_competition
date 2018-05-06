#!/usr/bin/env bash
name=$1
zone=$2
gcloud compute --project=ai-challenge-203314 instances create ${name} --zone=${zone} --machine-type=n1-standard-4 --subnet=default --maintenance-policy=MIGRATE --service-account=121469349162-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --min-cpu-platform=Automatic --image=centos-7-v20180401 --image-project=centos-cloud --boot-disk-size=40GB --boot-disk-type=pd-standard --boot-disk-device-name=${name}
