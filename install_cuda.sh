#!/usr/bin/env bash
echo \"Checking for CUDA and installing.\"
# Check for CUDA and try to install.
if ! rpm -q cuda-9-0; then
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-9.0.176-1.x86_64.rpm
  sudo rpm -i --force ./cuda-repo-rhel7-9.0.176-1.x86_64.rpm
  sudo yum clean all
  # Install Extra Packages for Enterprise Linux (EPEL) for dependencies
  sudo yum install epel-release -y
  sudo yum update -y
  sudo yum install cuda-9-0 -y
fi
# Verify that CUDA installed; retry if not.
if ! rpm -q cuda-9-0; then
  sudo yum install cuda-9-0 -y
fi
check_fail = $(nvidia-smi -pm 1 | grep fail)
# Enable persistence mode
if [ -z "$check_fail" ]; then
    reboot
fi