#!/usr/bin/env bash
zone=us-west1-b
name=instance-1

ip=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)

if [ -z "$ip" ]; then
    echo "Could not find instance, creating a new one"
    #./start_instance.sh ${name} ${zone}
    ip=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)
    echo "Instance ip: ${ip}"
    echo "Setting up instance: "
    echo "Sleeping during initial setup"
    ./setup_google.sh ${ip}
    echo "Done with setup, sleeping for 5mins to let everything finish"
    # sleep 5m
fi

#./train_google.sh $ip

if [ "$1" == "-d" ]; then
    if [ -z "$ip" ]; then
        echo "Didn't find instance to delete"
    else
        echo "Deleting Instance: "
        gcloud -q compute instances delete ${name} --zone=${zone}
        ssh-keygen -R ${ip}
    fi
fi