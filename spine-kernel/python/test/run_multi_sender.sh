#!/bin/bash

num_flow=${1-'3'}
cc=${2-'scubic'}
port=${3-'5001'}
# multiple sender
for i in $(seq 1 ${num_flow}); do
    iperf -c localhost -p $port -Z $cc -t 10 >flow-$i.log 2>&1 &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done
