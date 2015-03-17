#!/bin/bash

key_file=$HOME/.ssh/ec2.pem

declare -a ips=(
    # 127.0.0.1
    # 127.0.0.1
    # 127.0.0.1
    # 127.0.0.1
    # 127.0.0.1
    # 127.0.0.1
    # 127.0.0.1
    # 127.0.0.1
)

model=48
split=stratified
seed=101
for ip in ${ips[@]}
do
    echo "******* ${ip}"
    ssh -i ${key_file} ec2-user@${ip} "source /home/ec2-user/.zshrc; cd bowl ; nohup ./run.sh ${model} ${split} ${seed} </dev/null >./run.log 2>&1 &"
    seed=`expr ${seed} + 1`
done
