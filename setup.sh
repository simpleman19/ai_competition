sudo yum update
sudo yum install -y yum-utils git
sudo yum groupinstall -y development
sudo yum install -y https://centos7.iuscommunity.org/ius-release.rpm
sudo yum update
sudo yum install -y python36
sudo yum install -y python36-pip python36-devel
git clone https://github.com/simpleman19/ai_competition.git
cd ai_competition
python3.6 -m venv .env
. .env/bin/activate
pip install -r requirements.txt
