#!/usr/bin/env bash
uuid=$(uuidgen)
git tag ${uuid}
ssh 198.204.229.156 "cd ai_competition && git pull && git checkout ${uuid} && python3.6 train.py ${uuid}"