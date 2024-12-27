#include <getopt.h>
#include <signal.h>
#include <stdio.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "address.hh"
#include "common.hh"
#include "logging.hh"
#include "socket.hh"

#define BUFFER 1024

using namespace std;
using clock_type = std::chrono::high_resolution_clock;

std::chrono::_V2::system_clock::time_point ts_now = clock_type::now();
std::unique_ptr<std::ofstream> perf_log;
std::atomic<bool> recv_traffic(true);
std::atomic<size_t> recv_cnt = 0;
static size_t last_observed_recv_cnt = 0;

void signal_handler(int sig) {
  if (sig == SIGINT or sig == SIGKILL or sig == SIGTERM) {
    // terminate pyhelper
    recv_traffic = false;
    // close iperf
    if (perf_log) {
      perf_log->close();
    }
    // IPC socket will be closed later
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    exit(1);
  }
}

void perf_log_thread(const std::chrono::milliseconds interval) {
  // start regular congestion control parttern
  auto when_started = clock_type::now();
  auto target_time = when_started + interval;
  size_t tmp = 0;
  while (recv_traffic.load()) {
    // log the current throughput in Mbps
    tmp = recv_cnt;  // use this to avoid read/write competing
    unsigned long long current_thr =
        (tmp - last_observed_recv_cnt) * 8 / interval.count() * 1000 / 1000000;
    last_observed_recv_cnt = tmp;
    if (perf_log) {
      *perf_log << current_thr << endl;
    }
    std::this_thread::sleep_until(target_time);
    target_time += interval;
  }
}

void usage_error(const string& program_name) {
  cerr << "Usage: " << program_name << " [OPTION]... [COMMAND]" << endl;
  cerr << endl;
  cerr << "Options = --ip=IP_ADDR --port=PORT --cong=ALGORITHM (default: "
          "CUBIC) --perf-log=PATH(default is None) --perf-inteval=MS"
       << endl
       << "If perf_log is specified, the default log interval is 500ms" << endl;
  cerr << endl;

  throw runtime_error("invalid arguments");
}

int main(int argc, char** argv) {
  if (argc < 1) {
    usage_error(argv[0]);
  }
  const option command_line_options[] = {
      {"port", required_argument, nullptr, 'p'},
      {"cong", optional_argument, nullptr, 'c'},
      {"perf-log", optional_argument, nullptr, 'l'},
      {"perf-interval", optional_argument, nullptr, 'i'},
      {0, 0, nullptr, 0}};

  string service, cong_ctl, interval, perf_log_path;
  while (true) {
    const int opt = getopt_long(argc, argv, "", command_line_options, nullptr);
    if (opt == -1) { /* end of options */
      break;
    }
    switch (opt) {
    case 'c':
      cong_ctl = optarg;
      break;
    case 'i':
      interval = optarg;
      break;
    case 'l':
      perf_log_path = optarg;
      break;
    case 'p':
      service = optarg;
      break;
    case '?':
      usage_error(argv[0]);
      break;
    default:
      throw runtime_error("getopt_long: unexpected return value " +
                          to_string(opt));
    }
  }

  if (optind > argc) {
    usage_error(argv[0]);
  }

  /* default CC is cubic */
  if (cong_ctl.empty()) {
    cong_ctl = "cubic";
  }

  // init perf log file
  std::chrono::milliseconds log_interval(500ms);
  if (not perf_log_path.empty()) {
    perf_log.reset(new std::ofstream(perf_log_path));
    if (not perf_log->good()) {
      throw runtime_error(perf_log_path + ": error opening for writing");
    }
    if (not interval.empty()) {
      log_interval = std::chrono::milliseconds(stoi(interval));
    }
  }

  int port = stoi(service);
  // init server addr
  Address address("0.0.0.0", port);
  TCPSocket server;
  /* set reuse_addr */
  server.set_reuseaddr();
  server.bind(address);
  server.listen();
  LOG(INFO) << "Server listen at " << port;

  TCPSocket client = server.accept();
  client.set_congestion_control(cong_ctl);
  LOG(DEBUG) << "Congestion control algorithm: " << cong_ctl;

  // start logging thread
  thread log_thread;
  if (perf_log) {
    cerr << "Server start with perf logger" << endl;
    log_thread = std::move(std::thread(perf_log_thread, log_interval));
    *perf_log << "# Interval = " << log_interval.count() << "ms" << endl;
  }

  while (true) {
    recv_cnt += client.read(BUFFER).length();
  }
}