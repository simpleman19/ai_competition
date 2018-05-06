#!/usr/bin/env bash
zone=us-west1-b
name=training

ip=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)


if [ "$1" == "-d" ]; then
    if [ -z "$ip" ]; then
        echo "Didn't find instance to delete"

    else
        echo "Deleting Instance: "
        gcloud compute instances delete ${name} --zone=${zone}
    fi
else
    if [ -z "$ip" ]; then
        echo "Could not find instance, creating a new one"
        ./start_instance.sh ${name} ${zone}
        ip=$(gcloud compute instances describe ${name} --zone=${zone} | grep natIP: | sed 's/:/\n/g' | sed "s/ //g" | sed -n 2p)
        echo "Instance ip: ${ip}"
        echo "Setting up instance: "
        ./setup_google.sh ${ip}
    fi
    ./train_google.sh $ip
fi