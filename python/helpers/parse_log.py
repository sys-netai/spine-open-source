#!/usr/bin/env python

from os import path
import sys
import math
import itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse


class TunnelGraph(object):
    def __init__(
        self,
        tunnel_log,
        send_rate_path=None,
        throughput_path=None,
        capacity_path=None,
        delay_path=None,
        throughput_graph=None,
        delay_graph=None,
        ms_per_bin=500,
        flow_cc=None,
        summary_path=None,
    ):
        self.tunnel_log = tunnel_log
        self.throughput_graph = throughput_graph
        self.capacity_path = capacity_path
        self.delay_graph = delay_graph
        self.ms_per_bin = ms_per_bin
        self.send_rate_path = send_rate_path
        self.throughput_path = throughput_path
        self.delay_path = delay_path
        self.flow_cc = flow_cc
        self.summary_path = summary_path

    def ms_to_bin(self, ts, first_ts):
        return int((ts - first_ts) / self.ms_per_bin)

    def bin_to_s(self, bin_id):
        return bin_id * self.ms_per_bin / 1000.0

    def parse_tunnel_log(self):
        tunlog = open(self.tunnel_log)

        self.flows = {}
        first_ts = None
        capacities = {}

        arrivals = {}
        departures = {}
        self.delays_t = {}
        self.delays = {}
        # average delay in one bin
        self.delays_per_bin = {}

        first_capacity = None
        last_capacity = None
        first_arrival = {}
        last_arrival = {}
        first_departure = {}
        last_departure = {}

        total_first_departure = None
        total_last_departure = None
        total_arrivals = 0
        total_departures = 0

        while True:
            line = tunlog.readline()
            if not line:
                break

            if line.startswith("#"):
                continue

            items = line.split()
            ts = float(items[0])
            event_type = items[1]
            num_bits = int(items[2]) * 8

            if first_ts is None:
                first_ts = ts

            bin_id = self.ms_to_bin(ts, first_ts)

            if event_type == "#":
                capacities[bin_id] = capacities.get(bin_id, 0) + num_bits

                if first_capacity is None:
                    first_capacity = ts

                if last_capacity is None or ts > last_capacity:
                    last_capacity = ts
            elif event_type == "+":
                if len(items) == 4:
                    flow_id = int(items[-1])
                else:
                    flow_id = 0

                self.flows[flow_id] = True

                if flow_id not in arrivals:
                    arrivals[flow_id] = {}
                    first_arrival[flow_id] = ts

                if flow_id not in last_arrival:
                    last_arrival[flow_id] = ts
                else:
                    if ts > last_arrival[flow_id]:
                        last_arrival[flow_id] = ts

                old_value = arrivals[flow_id].get(bin_id, 0)
                arrivals[flow_id][bin_id] = old_value + num_bits

                total_arrivals += num_bits
            elif event_type == "-":
                if len(items) == 5:
                    flow_id = int(items[-1])
                else:
                    flow_id = 0

                self.flows[flow_id] = True

                if flow_id not in departures:
                    departures[flow_id] = {}
                    first_departure[flow_id] = ts

                if flow_id not in last_departure:
                    last_departure[flow_id] = ts
                else:
                    if ts > last_departure[flow_id]:
                        last_departure[flow_id] = ts

                old_value = departures[flow_id].get(bin_id, 0)
                departures[flow_id][bin_id] = old_value + num_bits

                total_departures += num_bits

                # update total variables
                if total_first_departure is None:
                    total_first_departure = ts
                if total_last_departure is None or ts > total_last_departure:
                    total_last_departure = ts

                # store delays in a list for each flow and sort later
                delay = float(items[3])
                if flow_id not in self.delays:
                    self.delays[flow_id] = []
                    self.delays_t[flow_id] = []
                self.delays[flow_id].append(delay)
                self.delays_t[flow_id].append((ts - first_ts) / 1000.0)

                if flow_id not in self.delays_per_bin:
                    self.delays_per_bin[flow_id] = {}
                if bin_id not in self.delays_per_bin[flow_id]:
                    self.delays_per_bin[flow_id][bin_id] = []
                self.delays_per_bin[flow_id][bin_id].append(delay)

        tunlog.close()

        us_per_bin = 1000.0 * self.ms_per_bin

        self.avg_capacity = None
        self.link_capacity = []
        self.link_capacity_t = []
        if capacities:
            # calculate average capacity
            if last_capacity == first_capacity:
                self.avg_capacity = 0
            else:
                delta = 1000.0 * (last_capacity - first_capacity)
                self.avg_capacity = sum(capacities.values()) / delta

            # transform capacities into a list
            capacity_bins = list(capacities.keys())
            for bin_id in range(min(capacity_bins), max(capacity_bins) + 1):
                self.link_capacity.append(capacities.get(bin_id, 0) / us_per_bin)
                self.link_capacity_t.append(self.bin_to_s(bin_id))

        # calculate ingress and egress throughput for each flow
        self.ingress_tput = {}
        self.egress_tput = {}
        self.ingress_t = {}
        self.egress_t = {}
        self.avg_ingress = {}
        self.avg_egress = {}
        self.percentile_delay = {}
        self.average_delay = {}
        self.loss_rate = {}
        # Xudong: Pantheon define reward log(throughput/rtt)
        self.reward = {}
        total_delays = []

        for flow_id in self.flows:
            self.ingress_tput[flow_id] = []
            self.egress_tput[flow_id] = []
            self.ingress_t[flow_id] = []
            self.egress_t[flow_id] = []
            self.avg_ingress[flow_id] = 0
            self.avg_egress[flow_id] = 0
            self.reward[flow_id] = []

            if flow_id in arrivals:
                # calculate average ingress and egress throughput
                first_arrival_ts = first_arrival[flow_id]
                last_arrival_ts = last_arrival[flow_id]

                if last_arrival_ts == first_arrival_ts:
                    self.avg_ingress[flow_id] = 0
                else:
                    delta = 1000.0 * (last_arrival_ts - first_arrival_ts)
                    flow_arrivals = sum(arrivals[flow_id].values())
                    self.avg_ingress[flow_id] = flow_arrivals / delta

                ingress_bins = list(arrivals[flow_id].keys())
                for bin_id in range(min(ingress_bins), max(ingress_bins) + 1):
                    self.ingress_tput[flow_id].append(
                        arrivals[flow_id].get(bin_id, 0) / us_per_bin
                    )
                    self.ingress_t[flow_id].append(self.bin_to_s(bin_id))

            if flow_id in departures:
                first_departure_ts = first_departure[flow_id]
                last_departure_ts = last_departure[flow_id]

                if last_departure_ts == first_departure_ts:
                    self.avg_egress[flow_id] = 0
                else:
                    delta = 1000.0 * (last_departure_ts - first_departure_ts)
                    flow_departures = sum(departures[flow_id].values())
                    self.avg_egress[flow_id] = flow_departures / delta

                egress_bins = list(departures[flow_id].keys())

                self.egress_tput[flow_id].append(0.0)
                self.egress_t[flow_id].append(self.bin_to_s(min(egress_bins)))

                for bin_id in range(min(egress_bins), max(egress_bins) + 1):
                    self.egress_tput[flow_id].append(
                        departures[flow_id].get(bin_id, 0) / us_per_bin
                    )
                    self.egress_t[flow_id].append(self.bin_to_s(bin_id + 1))

            delay_bins = list(self.delays_per_bin[flow_id].keys())

            # calculate 95th percentile per-packet one-way delay
            # Xudong: calculate average one way delay
            self.percentile_delay[flow_id] = None
            self.average_delay[flow_id] = None
            if flow_id in self.delays:
                self.percentile_delay[flow_id] = np.percentile(
                    self.delays[flow_id], 95, interpolation="nearest"
                )
                self.average_delay[flow_id] = np.mean(self.delays[flow_id])
                total_delays += self.delays[flow_id]

            # calculate loss rate for each flow
            if flow_id in arrivals and flow_id in departures:
                flow_arrivals = sum(arrivals[flow_id].values())
                flow_departures = sum(departures[flow_id].values())

                self.loss_rate[flow_id] = None
                if flow_arrivals > 0:
                    self.loss_rate[flow_id] = 1 - 1.0 * flow_departures / flow_arrivals

        self.total_loss_rate = None
        if total_arrivals > 0:
            self.total_loss_rate = 1 - 1.0 * total_departures / total_arrivals

        # calculate total average throughput and 95th percentile delay
        self.total_avg_egress = None
        if total_last_departure == total_first_departure:
            self.total_duration = 0
            self.total_avg_egress = 0
        else:
            self.total_duration = total_last_departure - total_first_departure
            self.total_avg_egress = total_departures / (1000.0 * self.total_duration)

        self.total_percentile_delay = None
        self.total_average_delay = None
        if total_delays:
            self.total_percentile_delay = np.percentile(
                total_delays, 95, interpolation="nearest"
            )
            self.total_average_delay = np.mean(total_delays)

    def flip(self, items, ncol):
        return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

    def plot_throughput_graph(self):
        empty_graph = True
        fig, ax = plt.subplots()

        if self.link_capacity:
            empty_graph = False
            ax.fill_between(
                self.link_capacity_t, 0, self.link_capacity, facecolor="linen"
            )
        # save link capactiy trace
        with open(self.capacity_path, "w", 1) as f:
            for i, j in zip(self.link_capacity_t, self.link_capacity):
                f.write("{}\t{}\n".format(i, j))

        colors = [
            "b",
            "g",
            "r",
            "y",
            "c",
            "m",
            "indigo",
            "lime",
            "chocolate",
            "darkorange",
        ]
        color_i = 0
        flow_cnt = 0
        for flow_id in self.flows:
            color = colors[color_i]
            in_label = "Flow %s ingress (mean %.2f Mbit/s)"
            e_label = "Flow %s egress (mean %.2f Mbit/s)"

            # custom_flow_label
            if self.flow_cc:
                in_label = self.flow_cc[flow_id - 1] + " " + in_label
                e_label = self.flow_cc[flow_id - 1] + " " + e_label

            if flow_id in self.ingress_tput and flow_id in self.ingress_t:
                empty_graph = False
                ax.plot(
                    self.ingress_t[flow_id],
                    self.ingress_tput[flow_id],
                    label=in_label % (flow_id, self.avg_ingress.get(flow_id, 0)),
                    color=color,
                    linestyle="dashed",
                )

            if flow_id in self.egress_tput and flow_id in self.egress_t:
                empty_graph = False
                ax.plot(
                    self.egress_t[flow_id],
                    self.egress_tput[flow_id],
                    label=e_label % (flow_id, self.avg_egress.get(flow_id, 0)),
                    color=color,
                )

            if flow_cnt == 0:
                # first flow, create the throughput and sending rate log file
                with open(self.send_rate_path, "w", 1) as f:
                    for i, j in zip(
                        self.ingress_t[flow_id], self.ingress_tput[flow_id]
                    ):
                        f.write("{}\t{}\t{}\n".format(flow_id, i, j))

                with open(self.throughput_path, "w", 1) as f:
                    for i, j in zip(self.egress_t[flow_id], self.egress_tput[flow_id]):
                        f.write("{}\t{}\t{}\n".format(flow_id, i, j))
                        
                with open(self.delay_path, "w", 1) as f:
                    tmp_delays = []
                    last_i = 0
                    for i, j in zip(self.delays_t[flow_id], self.delays[flow_id]):
                        if i//0.5 != last_i//0.5:
                            f.write("{}\t{}\t{}\n".format(flow_id, 0.5 * (last_i//0.5), np.mean(tmp_delays)  if len(tmp_delays)!= 0 else 0))
                            tmp_delays = []
                        tmp_delays.append(j)
                        last_i = i
                        
            else:
                # append the file
                with open(self.send_rate_path, "a+") as f:
                    for i, j in zip(
                        self.ingress_t[flow_id], self.ingress_tput[flow_id]
                    ):
                        f.write("{}\t{}\t{}\n".format(flow_id, i, j))

                with open(self.throughput_path, "a+") as f:
                    for i, j in zip(self.egress_t[flow_id], self.egress_tput[flow_id]):
                        f.write("{}\t{}\t{}\n".format(flow_id, i, j))

                with open(self.delay_path, "a+") as f:
                    for i, j in zip(self.delays_t[flow_id], self.delays[flow_id]):
                        if i//0.5 != last_i//0.5:
                            f.write("{}\t{}\t{}\n".format(flow_id, 0.5 * (last_i//0.5), np.mean(tmp_delays)  if len(tmp_delays)!= 0 else 0))
                            tmp_delays = []
                        tmp_delays.append(j)
                        last_i = i

            color_i += 1
            flow_cnt += 1
            if color_i == len(colors):
                color_i = 0

        if empty_graph:
            sys.stderr.write("No valid throughput graph is generated\n")
            return

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Throughput (Mbit/s)", fontsize=12)

        if self.link_capacity and self.avg_capacity:
            ax.set_title(
                "Average capacity %.2f Mbit/s (shaded region)" % self.avg_capacity
            )

        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
            self.flip(handles, 2),
            self.flip(labels, 2),
            scatterpoints=1,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=2,
            fontsize=12,
        )

        fig.set_size_inches(12, 6)
        fig.savefig(
            self.throughput_graph,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
            pad_inches=0.2,
        )

    def plot_delay_graph(self):
        empty_graph = True
        fig, ax = plt.subplots()

        max_delay = 0
        colors = ["b", "g", "r", "y", "c", "m"]
        color_i = 0
        for flow_id in self.flows:
            color = colors[color_i]
            if flow_id in self.delays and flow_id in self.delays_t:
                empty_graph = False
                max_delay = max(max_delay, max(self.delays_t[flow_id]))

                ax.scatter(
                    self.delays_t[flow_id],
                    self.delays[flow_id],
                    s=1,
                    color=color,
                    marker=".",
                    #    label='Flow %s (95th percentile %.2f ms)'
                    label="Flow %s (average %.2f ms)"
                    % (flow_id, self.average_delay.get(flow_id, 0)),
                )
                color_i += 1
                if color_i == len(colors):
                    color_i = 0

        if empty_graph:
            sys.stderr.write("No valid delay graph is generated\n")
            return

        ax.set_xlim(0, int(math.ceil(max_delay)))
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Per-packet one-way delay (ms)", fontsize=12)

        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
            self.flip(handles, 3),
            self.flip(labels, 3),
            scatterpoints=1,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=3,
            fontsize=12,
            markerscale=5,
            handletextpad=0,
        )

        fig.set_size_inches(12, 6)
        fig.savefig(
            self.delay_graph,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
            pad_inches=0.2,
        )

    def statistics_string(self):
        if len(self.flows) == 1:
            flows_str = "flow"
        else:
            flows_str = "flows"
        ret = "-- Total of %d %s:\n" % (len(self.flows), flows_str)

        if self.avg_capacity is not None:
            ret += "Average capacity: %.2f Mbit/s\n" % self.avg_capacity

        if self.total_avg_egress is not None:
            ret += "Average throughput: %.2f Mbit/s" % self.total_avg_egress

        if self.avg_capacity is not None and self.total_avg_egress is not None:
            ret += " (%.1f%% utilization)" % (
                100.0 * self.total_avg_egress / self.avg_capacity
            )
        ret += "\n"

        if self.total_average_delay is not None:
            # ret += ('95th percentile per-packet one-way delay: %.3f ms\n' %
            ret += (
                "Average per-packet one-way delay: %.3f ms\n" % self.total_average_delay
            )

        if self.total_loss_rate is not None:
            ret += "Loss rate: %.2f%%\n" % (self.total_loss_rate * 100.0)

        # prepare the stat_str, we want to flow_id being sorted
        for flow_id in sorted(self.flows):
            ret += "-- Flow %d:\n" % flow_id

            if flow_id in self.avg_egress and self.avg_egress[flow_id] is not None:
                ret += "Average throughput: %.2f Mbit/s\n" % self.avg_egress[flow_id]

            if (
                flow_id in self.average_delay
                and self.average_delay[flow_id] is not None
            ):
                # ret += ('95th percentile per-packet one-way delay: %.3f ms\n' %
                ret += (
                    "Average per-packet one-way delay: %.3f ms\n"
                    % self.average_delay[flow_id]
                )

            if flow_id in self.loss_rate and self.loss_rate[flow_id] is not None:
                ret += "Loss rate: %.2f%%\n" % (self.loss_rate[flow_id] * 100.0)

        return ret

    def run(self):
        self.parse_tunnel_log()

        if self.throughput_graph:
            self.plot_throughput_graph()

        if self.delay_graph:
            self.plot_delay_graph()

        plt.close("all")

        tunnel_results = {}
        tunnel_results["throughput"] = self.total_avg_egress
        tunnel_results["delay"] = self.total_average_delay
        tunnel_results["percentile_delay"] = self.total_percentile_delay
        tunnel_results["loss"] = self.total_loss_rate
        tunnel_results["duration"] = self.total_duration
        tunnel_results["stats"] = self.statistics_string()

        flow_data = {}
        flow_data["all"] = {}
        flow_data["all"]["tput"] = self.total_avg_egress
        flow_data["all"]["capacity"] = self.avg_capacity
        flow_data["all"]["utilization"] = self.total_avg_egress / self.avg_capacity
        flow_data["all"]["delay"] = self.total_average_delay
        flow_data["all"]["percentile_delay"] = self.total_percentile_delay
        flow_data["all"]["loss"] = self.total_loss_rate

        for flow_id in self.flows:
            if flow_id != 0:
                flow_data[flow_id] = {}
                if self.flow_cc:
                    flow_data[flow_id]["cc"] = self.flow_cc[flow_id - 1]
                flow_data[flow_id]["tput"] = self.avg_egress[flow_id]
                flow_data[flow_id]["delay"] = self.average_delay[flow_id]
                flow_data[flow_id]["percentile_delay"] = self.percentile_delay[flow_id]
                flow_data[flow_id]["loss"] = self.loss_rate[flow_id]

        tunnel_results["flow_data"] = flow_data

        return tunnel_results


def parse_tunnel_graph():
    parser = argparse.ArgumentParser(
        description="evaluate throughput and delay of a tunnel log and "
        "generate graphs"
    )

    parser.add_argument("tunnel_log", metavar="tunnel-log", help="tunnel log file")
    parser.add_argument(
        "--throughput",
        metavar="OUTPUT-GRAPH",
        action="store",
        dest="throughput_graph",
        help="throughput graph to save as (default None)",
    )
    parser.add_argument(
        "--delay",
        metavar="OUTPUT-GRAPH",
        action="store",
        dest="delay_graph",
        help="delay graph to save as (default None)",
    )
    parser.add_argument(
        "--ms-per-bin",
        metavar="MS-PER-BIN",
        type=int,
        default=500,
        help="bin size in ms (default 500)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_tunnel_graph()

    tunnel_graph = TunnelGraph(
        tunnel_log=args.tunnel_log,
        throughput_graph=args.throughput_graph,
        delay_graph=args.delay_graph,
        ms_per_bin=args.ms_per_bin,
    )
    tunnel_results = tunnel_graph.run()

    sys.stderr.write(tunnel_results["stats"])


if __name__ == "__main__":
    main()
