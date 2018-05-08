#!/usr/bin/env bash
echo \"Installing CUDA.\"
curl -O http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-8.0.61-1.x86_64.rpm
sudo rpm -i --force ./cuda-repo-rhel7-8.0.61-1.x86_64.rpm
sudo yum clean all
# Install Extra Packages for Enterprise Linux (EPEL) for dependencies
sudo yum install epel-release -y
sudo yum update -y
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
sudo yum -y install gcc gcc-c++ python-pip python-devel atlas atlas-devel gcc-gfortran openssl-devel libffi-devel
sudo yum install cuda -y
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo sh cuda_8.0.61_375.26_linux.run
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.3/prod/8.0_20180414/cudnn-8.0-linux-x64-v7.1
tar zxf cudnn-8.0-linux-x64-v7.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/.
sudo chmod a+x /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
echo "export PATH=/usr/local/cuda-8.0/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
check_fail=$(nvidia-smi -pm 1 | grep fail)
# Enable persistence mode
if [ ! -z "$check_fail" ]; then
    reboot
fi