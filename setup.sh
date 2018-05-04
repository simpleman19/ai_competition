#!/usr/bin/env bash
sudo yum update
sudo yum install -y yum-utils git nfs-utils
sudo yum groupinstall -y development
sudo yum install -y https://centos7.iuscommunity.org/ius-release.rpm
sudo yum update
sudo yum install -y python36
sudo yum install -y python36-pip python36-devel
git clone https://github.com/simpleman19/ai_competition.git
cd ai_competition
mkdir rf_data
sudo mount -t nfs 10.0.0.3:/media/share rf_data
python3.6 -m venv .env
. .env/bin/activate
pip install -r requirements.txt
