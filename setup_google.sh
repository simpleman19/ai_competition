#!/usr/bin/env bash
host=$1
ssh -i ~/.ssh/google_compute_engine chancert413_gmail_com@${host} bash -c "'
sudo tee /etc/yum.repos.d/gcsfuse.repo > /dev/null <<EOF
[gcsfuse]
name=gcsfuse (packages.cloud.google.com)
baseurl=https://packages.cloud.google.com/yum/repos/gcsfuse-el7-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg
       https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF
sudo yum update -y
sudo yum install -y yum-utils git nfs-utils gcsfuse rsync
sudo yum groupinstall -y development
sudo yum install -y https://centos7.iuscommunity.org/ius-release.rpm
sudo yum update
sudo yum install -y python36
sudo yum install -y python36-pip python36-devel python36-tkinter
git clone https://github.com/simpleman19/ai_competition.git
cd ai_competition
mkdir rf_data
mkdir temp
gcsfuse training_data_0 temp
rsync -avh temp/* rf_data/
fusermount -u temp
python3.6 -m venv .env
. .env/bin/activate
pip install -r requirements.txt
chmod +x install_cuda.sh
./install_cuda.sh
'"