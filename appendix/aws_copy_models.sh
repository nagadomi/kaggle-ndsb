#!/bin/bash

key_file=$HOME/.ssh/ec2.pem

declare -a ips=(
     #127.0.0.1
     #127.0.0.1
     #127.0.0.1
     #127.0.0.1
     #127.0.0.1
     #127.0.0.1
     #127.0.0.1
     #127.0.0.1
)

model=48
split=stratified
seed=101

for ip in ${ips[@]}
do
    scp -i ${key_file} -Cp ec2-user@${ip}:bowl/models/submission_${model}x_${split}_${seed}.txt .
    scp -i ${key_file} -Cp ec2-user@${ip}:bowl/models/cnn_${model}x_${split}_${seed}.t7 .
done

