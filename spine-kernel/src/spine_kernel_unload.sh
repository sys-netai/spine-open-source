#!/bin/bash

module="scubic"

if [ "$EUID" -ne 0 ]; then 
    echo "error: must run as root"
    usage
    exit 1
fi

# invoke rmmod with all arguments we got
/sbin/rmmod $module $* || exit 1