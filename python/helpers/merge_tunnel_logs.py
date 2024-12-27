import sys
import argparse
import heapq


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='single or multiple mode',
                                       dest='mode')

    # subparser for single mode
    single_parser = subparsers.add_parser(
        'single', help='merge the ingress log and egress log of a single '
        'tunnel into a "tunnel log"')
    single_parser.add_argument(
        '-i', action='store', metavar='INGRESS-LOG', dest='ingress_log',
        required=True, help='ingress log of a tunnel')
    single_parser.add_argument(
        '-e', action='store', metavar='EGRESS-LOG', dest='egress_log',
        required=True, help='egress log of a tunnel')
    single_parser.add_argument(
        '-o', action='store', metavar='OUTPUT-LOG', dest='output_log',
        required=True, help='tunnel log after merging')
    single_parser.add_argument(
        '-i-clock-offset', metavar='MS', type=float,
        help='clock offset on the end where ingress log is saved')
    single_parser.add_argument(
        '-e-clock-offset', metavar='MS', type=float,
        help='clock offset on the end where egress log is saved')

    # subparser for multiple mode
    multiple_parser = subparsers.add_parser(
        'multiple', help='merge the tunnel logs of one or more tunnels')
    multiple_parser.add_argument(
        '--link-log', action='store', metavar='LINK-LOG', dest='link_log',
        help='uplink or downlink log generated by mm-link')
    multiple_parser.add_argument(
        'tunnel_logs', metavar='TUNNEL-LOG', nargs='+',
        help='one or more tunnel logs generated by single mode')
    multiple_parser.add_argument(
        '-o', action='store', metavar='OUTPUT-LOG', dest='output_log',
        required=True, help='output log after merging')

    return parser.parse_args()


def parse_line(line):
    (ts, uid, size) = line.split('-')
    return (float(ts), int(uid), int(size))


def single_mode(args):
    recv_log = open(args.ingress_log)
    send_log = open(args.egress_log)
    output_log = open(args.output_log, 'w')

    # retrieve initial timestamp of sender from the first line
    line = send_log.readline()
    if not line:
        sys.exit('Warning: egress log is empty\n')

    send_init_ts = float(line.rsplit(':', 1)[-1])
    if args.e_clock_offset is not None:
        send_init_ts += args.e_clock_offset

    min_init_ts = send_init_ts

    # retrieve initial timestamp of receiver from the first line
    line = recv_log.readline()
    if not line:
        sys.exit('Warning: ingress log is empty\n')

    recv_init_ts = float(line.rsplit(':', 1)[-1])
    if args.i_clock_offset is not None:
        recv_init_ts += args.i_clock_offset

    if recv_init_ts < min_init_ts:
        min_init_ts = recv_init_ts

    output_log.write('# init timestamp: %.3f\n' % min_init_ts)

    # timestamp calibration to ensure non-negative timestamps
    send_cal = send_init_ts - min_init_ts
    recv_cal = recv_init_ts - min_init_ts

    # construct a hash table using uid as keys
    send_pkts = {}
    for line in send_log:
        (send_ts, send_uid, send_size) = parse_line(line)
        send_pkts[send_uid] = (send_ts + send_cal, send_size)

    send_log.seek(0)
    send_log.readline()

    # merge two sorted logs into one
    send_l = send_log.readline()
    if send_l:
        (send_ts, send_uid, send_size) = parse_line(send_l)

    recv_l = recv_log.readline()
    if recv_l:
        (recv_ts, recv_uid, recv_size) = parse_line(recv_l)

    while send_l or recv_l:
        if send_l:
            send_ts_cal = send_ts + send_cal
        if recv_l:
            recv_ts_cal = recv_ts + recv_cal

        if (send_l and recv_l and send_ts_cal <= recv_ts_cal) or not recv_l:
            output_log.write('%.3f + %s\n' % (send_ts_cal, send_size))
            send_l = send_log.readline()
            if send_l:
                (send_ts, send_uid, send_size) = parse_line(send_l)
        elif (send_l and recv_l and send_ts_cal > recv_ts_cal) or not send_l:
            if recv_uid in send_pkts:
                (paired_send_ts, paired_send_size) = send_pkts[recv_uid]
                # inconsistent packet size
                if paired_send_size != recv_size:
                    sys.exit(
                        'Warning: packet %s came into tunnel with size %s '
                        'but left with size %s\n' %
                        (recv_uid, paired_send_size, recv_size))
            else:
                # nonexistent packet
                sys.exit('Warning: received a packet with nonexistent '
                         'uid %s\n' % recv_uid)

            delay = recv_ts_cal - paired_send_ts
            output_log.write('%.3f - %s %.3f\n'
                             % (recv_ts_cal, recv_size, delay))
            recv_l = recv_log.readline()
            if recv_l:
                (recv_ts, recv_uid, recv_size) = parse_line(recv_l)

    recv_log.close()
    send_log.close()
    output_log.close()


def push_to_heap(heap, index, log_file, init_ts_delta):
    line = None

    while True:
        line = log_file.readline()
        if not line:
            break

        if line.startswith('#'):
            continue

        # if log_file is mm-link-log
        if index == -1:
            # find the next delivery opportunity
            if '#' in line:
                break
        else:
            break

    if line:
        line_list = line.strip().split()
        calibrated_ts = float(line_list[0]) + init_ts_delta

        line_list[0] = '%.3f' % calibrated_ts
        if line_list[1] == '#':
            line_list[2] = str(int(line_list[2]) - 4)
        line = ' '.join(line_list)
        heapq.heappush(heap, (calibrated_ts, index, line))

    return line


def multiple_mode(args):
    # open log files
    link_log = None
    if args.link_log:
        link_log = open(args.link_log)

    tun_logs = []
    for tun_log_name in args.tunnel_logs:
        tun_logs.append(open(tun_log_name))

    output_log = open(args.output_log, 'w')

    # maintain a min heap to merge sorted logs
    heap = []
    if link_log:
        # find initial timestamp in the mm-link log
        while True:
            line = link_log.readline()
            if not line:
                sys.exit('Warning: link log %s is empty' % link_log.name)

            if not line.startswith('# init timestamp'):
                continue

            link_init_ts = float(line.split(':')[1])
            min_init_ts = link_init_ts
            break
    else:
        min_init_ts = 1e20

    # find the smallest initial timestamp
    init_ts_delta = []
    for tun_log in tun_logs:
        while True:
            line = tun_log.readline()
            if not line:
                sys.exit('Warning: tunnel log %s is empty' % tun_log.name)

            if not line.startswith('# init timestamp'):
                continue

            init_ts = float(line.split(':')[1])
            init_ts_delta.append(init_ts)
            if init_ts < min_init_ts:
                min_init_ts = init_ts
            break

    if link_log:
        link_init_ts_delta = link_init_ts - min_init_ts

    for i in range(len(init_ts_delta)):
        init_ts_delta[i] -= min_init_ts

    output_log.write('# init timestamp: %.3f\n' % min_init_ts)

    # build the min heap
    if link_log:
        line = push_to_heap(heap, -1, link_log, link_init_ts_delta)
        if not line:
            sys.exit('Warning: no delivery opportunities found\n')

    for i in range(len(tun_logs)):
        line = push_to_heap(heap, i, tun_logs[i], init_ts_delta[i])
        if not line:
            sys.exit(
                'Warning: %s does not contain any arrival or '
                'departure events\n' % tun_logs[i].name)

    # merge all log files
    while heap:
        (ts, index, line) = heapq.heappop(heap)

        # append flow ids to arrival and departure events
        if index != -1:
            line += ' %s' % (index + 1)

        output_log.write(line + '\n')

        if index == -1:
            push_to_heap(heap, index, link_log, link_init_ts_delta)
        else:
            push_to_heap(heap, index, tun_logs[index], init_ts_delta[index])

    # close log files
    if link_log:
        link_log.close()
    for tun_log in tun_logs:
        tun_log.close()
    output_log.close()


def main():
    args = parse_arguments()

    if args.mode == 'single':
        single_mode(args)
    else:
        multiple_mode(args)


if __name__ == '__main__':
    main()
