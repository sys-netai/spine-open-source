#!/bin/bash

module="scubic"
mode="664"

if [ "$EUID" -ne 0 ]; then 
    echo "error: must run as root"
    exit 1
fi

if [ -z "lsmod | grep 'ccp'" ]; then
    echo "$(module) kernel module already loaded"
    exit 1
fi

# invoke insmod with all arguments we got
# and use a pathname, as insmod doesn't look in . by default
/sbin/insmod ./$module.ko || ((dmesg | tail) && exit 1)


ALLOWED=$(sudo cat /proc/sys/net/ipv4/tcp_allowed_congestion_control)
echo "${ALLOWED} scubic" | sudo tee /proc/sys/net/ipv4/tcp_allowed_congestion_control
