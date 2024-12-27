#include <arpa/inet.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "address.hh"
#include "common.hh"
#include "exception.hh"
#include "filesystem.hh"
#include "ipc_socket.hh"
#include "json.hpp"
#include "logging.hh"
#include "serialization.hh"
#include "socket.hh"

struct tcp_orca_info {
  u32 min_rtt;       /* min-filtered RTT in uSec */
  u32 avg_urtt;      /* averaged RTT in uSec from the previous info request till
                        now*/
  u32 cnt;           /* number of RTT samples uSed for averaging */
  unsigned long thr; /*Bytes per second*/
  u32 thr_cnt;
  u32 cwnd;
  u32 pacing_rate;
  u32 lost_bytes;
  u32 srtt_us;         /* smoothed round trip time << 3 in usecs */
  u32 snd_ssthresh;    /* Slow start size threshold*/
  u32 packets_out;     /* Packets which are "in flight"*/
  u32 retrans_out;     /* Retransmitted packets out*/
  u32 max_packets_out; /* max packets_out in last window */
  u32 mss;

  void init() {
    min_rtt = 0;
    avg_urtt = 0;
    cnt = 0;
    thr = 0;
    thr_cnt = 0;
    cwnd = 0;
    pacing_rate = 0;
    lost_bytes = 0;
    srtt_us = 0;
    snd_ssthresh = 0;
    retrans_out = 0;
    max_packets_out = 0;
    mss = 0;
  }
  tcp_orca_info& operator=(const tcp_orca_info& a) {
    this->min_rtt = a.min_rtt;
    this->avg_urtt = a.avg_urtt;
    this->cnt = a.cnt;
    this->thr = a.thr;
    this->thr_cnt = a.thr_cnt;
    this->cwnd = a.cwnd;
    this->pacing_rate = a.pacing_rate;
    this->lost_bytes = a.lost_bytes;
    this->snd_ssthresh = a.snd_ssthresh;
    this->packets_out = a.packets_out;
    this->retrans_out = a.retrans_out;
    this->max_packets_out = a.max_packets_out;
    this->mss = a.mss;
  }
} orca_info;

// test set orca info
using namespace std;
using namespace std::literals;
using clock_type = std::chrono::high_resolution_clock;

// short name
using json = nlohmann::json;
using IPC_ptr = std::unique_ptr<FileDescriptor>;

bool send_traffic = true;

struct TCPDeepCCInfo info;

bool check_prefix(const string& s, vector<string>& results) {
  stringstream os(s);
  string substr;
  // prefix should be ACTION
  getline(os, substr, ':');
  if (substr != "ACTION") return false;
  while (os.good()) {
    // get first string delimited by comma
    getline(os, substr, ':');
    results.push_back(substr);
  }
  return true;
}

std::unique_ptr<FileDescriptor> setup_ipc_socket(const string& ipc_file) {
  if (fs::exists(ipc_file) && fs::is_socket(ipc_file))
    throw runtime_error("IPC file exists but it is not a socket file");
  IPCSocket ipc;
  ipc.bind(ipc_file);
  ipc.listen();
  LOG(INFO) << "Sender IPC listen at " << ipc_file;
  return std::make_unique<FileDescriptor>(ipc.accept());
}

int get_orca_info(int sk, struct tcp_orca_info* info) {
  int tcp_info_length = sizeof(*info);

  int ret = getsockopt(sk, 6, TCP_DEEPCC_INFO, (void*)info,
                       (socklen_t*)&tcp_info_length);

  if (ret < 0) {
    LOG(ERROR) << "ERROR " << strerror(errno) << " on get_orca_info";
  }
  return ret;
};

void do_congestion_control(const int sock, IPC_ptr& ipc_sock) {
  orca_info.avg_urtt = -1;
  orca_info.thr_cnt = -1;
  orca_info.thr = -1;
  get_orca_info(sock, &orca_info);
  // SystemCall("getsockopt get deepCC info",
  //            getsockopt(sock, IPPROTO_TCP, TCP_ORCA_INFO, (void*)info,
  //                       (socklen_t*)&tcp_info_length));
  fprintf(stdout,
          "Get orca info. avg_thr: %d, thr_cnt: %d, avg_urtt: %d, min_rtt: %d, "
          "cnt: %d\n",
          orca_info.thr, orca_info.thr_cnt, orca_info.avg_urtt,
          orca_info.min_rtt, orca_info.cnt);
  // LOG(DEBUG) << "TCPDeepCCInfo" << info.to_string();
  /*
   * write info to IPC socket
   * info should be string dumped from json
   */
  // json state = info.to_json();
  // uint16_t len = state.dump().length();
  // ipc_sock->write(put_field(len) + state.dump());

  // /* read feedback, feedback shall come from IPC socket */
  // auto header = ipc_sock->read_exactly(2);
  // auto action_len = get_uint16(header.data());
  // auto action = ipc_sock->read_exactly(action_len);

  // int cwnd = json::parse(action).at("cwnd");
  int cwnd = 50;
  try {
    LOG(DEBUG) << "GET cwnd from user: " << cwnd;
    SystemCall("setsockopt set_cwnd",
               setsockopt(sock, IPPROTO_TCP, TCP_CWND, (void*)&cwnd,
                          socklen_t(sizeof(cwnd))));
  } catch (const exception& e) {
    print_exception("set_cwnd", e);
    throw runtime_error("Cannot set tcp_cwnd: " + cwnd);
  }
}

void control_thread(const int sock, IPC_ptr& ipc_sock,
                    const std::chrono::milliseconds interval) {
  auto when_started = clock_type::now();
  auto target_time = when_started + interval;
  while (send_traffic) {
    do_congestion_control(sock, ipc_sock);
    std::this_thread::sleep_until(target_time);
    target_time += interval;
  }
}

void signal_handler(int sig) {
  if (sig == SIGINT or sig == SIGKILL or sig == SIGTERM) {
    send_traffic = false;
    LOG(INFO) << "Caught signal, exiting...";
    exit(1);
  }
}

void data_thread(const int sock) {
  int file = open("/tmp/large-file.txt", O_RDONLY);
  int buf_len = 8194;
  char data[buf_len];
  while (send_traffic) {
    int len = read(file, (void*)&data, buf_len);
    if (len > 0) {
      write(sock, data, buf_len);
    } else
      break;
  }
}

void usage_error(const string& program_name) {
  cerr << "Usage: " << program_name << " [OPTION]... [COMMAND]" << endl;
  cerr << endl;
  cerr << "Options = --ip=IP_ADDR --port=PORT --cong=ALGORITHM --ipc=IPC_FILE "
          "--interval=INTERVAL (Milliseconds)"
       << endl;
  cerr << endl;
  cerr << "Default congestion control algorithms for incoming TCP is CUBIC"
       << "Default control interval is 10ms" << endl;

  throw runtime_error("invalid arguments");
}

int main(int argc, char** argv) {
  /* register signal handler */
  signal(SIGTERM, signal_handler);
  signal(SIGKILL, signal_handler);
  signal(SIGINT, signal_handler);

  if (argc < 1) {
    usage_error(argv[0]);
  }
  const option command_line_options[] = {
      {"ip", required_argument, nullptr, 'a'},
      {"port", required_argument, nullptr, 'p'},
      {"ipc", optional_argument, nullptr, 'i'},
      {"cong", optional_argument, nullptr, 'c'},
      {"interval", optional_argument, nullptr, 't'},
      {0, 0, nullptr, 0}};

  string ip, service, cong_ctl, ipc_file, interval;
  while (true) {
    const int opt = getopt_long(argc, argv, "", command_line_options, nullptr);
    if (opt == -1) { /* end of options */
      break;
    }
    switch (opt) {
    case 'a':
      ip = optarg;
      break;
    case 'c':
      cong_ctl = optarg;
      break;
    case 'i':
      ipc_file = optarg;
      break;
    case 'p':
      service = optarg;
      break;
    case 't':
      interval = optarg;
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

  /* setup IPC */
  IPC_ptr ipc = nullptr;
  std::chrono::milliseconds control_interval(100ms);
  if (not ipc_file.empty()) {
    ipc = std::move(setup_ipc_socket(ipc_file));
    if (not interval.empty()) {
      control_interval = std::move(std::chrono::milliseconds(stoi(interval)));
      LOG(INFO) << "IPC with env has been established, control interval is "
                << control_interval.count() << "ms";
    }
  }

  /* default CC is cubic */
  if (cong_ctl.empty()) {
    cong_ctl = "cubic";
  }

  int port = stoi(service);
  /* init server addr */
  sockaddr_in serveraddr;
  memset(&serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  // IP address
  serveraddr.sin_addr.s_addr = inet_addr(ip.c_str());
  // Port number
  serveraddr.sin_port = htons(port);

  int client;
  client = SystemCall("socket", socket(AF_INET, SOCK_STREAM, 0));
  int reuse = 1;

  /* set reuse_addr */
  SystemCall("setsockopt reuseaddr",
             setsockopt(client, SOL_SOCKET, SO_REUSEADDR, (void*)&reuse,
                        socklen_t(sizeof(reuse))));

  /* set no delay */
  SystemCall("setsockopt no_delay", setsockopt(client, IPPROTO_TCP, TCP_NODELAY,
                                               &reuse, sizeof(reuse)));
  /* set congestion control */
  SystemCall(
      "setsockopt set cong_ctl",
      setsockopt(client, IPPROTO_TCP, TCP_CONGESTION, (void*)cong_ctl.c_str(),
                 socklen_t(strlen(cong_ctl.c_str()))));
  LOG(INFO) << "Client set congestion control to " << cong_ctl;
  // /* enable orca-plugin */
  // int enable_orca = 4;
  // SystemCall("setsockopt enable_deepcc",
  //            setsockopt(client, IPPROTO_TCP, TCP_ORCA_ENABLE,
  //                       (void*)&enable_orca,
  //                       socklen_t(sizeof(enable_orca))));

  LOG(INFO) << "Client enables deepCC plugin: 2";
  SystemCall("connect", connect(client, (struct sockaddr*)&serveraddr,
                                socklen_t(sizeof(sockaddr))));

  LOG(INFO) << "Connected to server, socket fd is " << client;
  /* start data thread and control thread */
  // !!! enable orca should be set after socket being connected
  int enable_orca = 2;
  if (setsockopt(client, IPPROTO_TCP, TCP_DEEPCC_ENABLE, &enable_orca,
                 sizeof(enable_orca)) < 0) {
    exit(-1);
  }
  thread ct;
  if (ipc != nullptr) {
    ct = std::move(
        thread(control_thread, client, std::ref(ipc), control_interval));
    LOG(INFO) << "Started control thread ... ";
  }
  thread dt(data_thread, client);

  /* wait for finish */
  LOG(INFO) << "Client is sending data ... ";
  dt.join();
}
