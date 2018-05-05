#!/usr/bin/env bash
uuid=$(uuidgen)
git commit -am "Auto commit - Add model to train list"
git push
git tag ${uuid}
git push origin --tags
echo "${uuid}" >> train_list