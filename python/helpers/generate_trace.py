import argparse
import numpy as np
from os import path
from random import random

MTU = 1500 * 8
LENGTH = 500
GRADULARITY = 100 #milliseconds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bandwidth', metavar='Mbps', required=True,
                        help='constant bandwidth (Mbps)')
    parser.add_argument('--output-dir', metavar='DIR', required=True,
                        help='directory to output trace')
    parser.add_argument('--var-freq', required=False, type=float, default = 0,
                        help='variance frequency')
    parser.add_argument('--var-range', required=False, type=float, default = 0,
                        help='variance range')
    args = parser.parse_args()

    trace_path = path.join(args.output_dir, '%smbps.trace' % args.bandwidth)
    generate_trace_file(trace_path, args.bandwidth, args.var_freq, args.var_range)

def get_bandwidth_list(input_path):
    count = 0
    last_number = 0
    bandwidth_list = []
    with open(input_path, "r") as trace:
        for line in trace.readlines():
            number = int(line)
            # print("line:", number)
            count += 1
            if (last_number + 1) % GRADULARITY == 0 and number != last_number:
                bw = count * MTU / GRADULARITY / 1e3
                bandwidth_list.append(bw)
                count = 0
            last_number = number
    return bandwidth_list
        

def generate_trace_file(output_path, bw, var_freq, var_range):
    # number of packets in 60*5 seconds
    # num_packets = int(float(args.bandwidth) * 5000 * 5)
    
    # trace path
    # make_sure_path_exists(args.output_dir)

    # write timestamps to trace
    capacity = float(bw)
    last_bw = capacity
    bandwidth_list = []
    count = 0
    acc_bandwidth = 0.
    with open(output_path, 'w') as trace:
        for i in range(LENGTH*1000):    
            count += 1
            acc_bandwidth += last_bw
            if count == GRADULARITY:
                mean_bw = acc_bandwidth / count
                bandwidth_list.append(mean_bw)
                ts_list = np.linspace(0, GRADULARITY, num=int(acc_bandwidth * 1e3 / MTU), endpoint=False)
                for ts in ts_list:
                    trace.write('%d\n' % (i+1-GRADULARITY+ts))
                acc_bandwidth = 0.
                count = 0
            if random() < var_freq:
                var = np.random.normal(loc=1, scale=var_range)
                var = np.clip(var, 0.8, 1.2)
                last_bw = max(min(var * last_bw, 1.5 * capacity), 0.5 * capacity)
    return bandwidth_list # 100 * LRNGTH entries


if __name__ == '__main__':
    main()
