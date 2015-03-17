#!/bin/bash

command=$@
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
if test $# -eq 0; then
    echo "error: command not specified"
    exit
fi

for ip in ${ips[@]}
do
    echo "******* ${ip}"
    ssh -i ${key_file} ec2-user@${ip} source /home/ec2-user/.zshrc\; ${command}
done
