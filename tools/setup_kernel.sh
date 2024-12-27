#!/bin/bash

sudo sysctl -w net.core.default_qdisc="fq"
sudo sysctl -w net.core.rmem_default="16777216"
sudo sysctl -w net.core.rmem_max="536870912"
sudo sysctl -w net.core.wmem_default="16777216"
sudo sysctl -w net.core.wmem_max="536870912"
sudo sysctl -w net.ipv4.tcp_rmem="4096 16777216 536870912"
sudo sysctl -w net.ipv4.tcp_wmem="4096 16777216 536870912"

sudo sysctl -w net.ipv4.tcp_autocorking=0
sudo sysctl -w net.ipv4.tcp_no_metrics_save=1
sudo sysctl -w net.ipv4.ip_forward=1
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=524288