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
#include "deepcc_socket.hh"
#include "exception.hh"
#include "filesystem.hh"
#include "ipc_socket.hh"
#include <nlohmann/json.hpp>
#include "logging.hh"
#include "serialization.hh"
#include "socket.hh"
#include "tcp_info.hh"

using namespace std;
using namespace std::literals;
using clock_type = std::chrono::high_resolution_clock;

// short name
using json = nlohmann::json;
using IPC_ptr = std::unique_ptr<FileDescriptor>;

bool send_traffic = true;

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

void do_congestion_control(DeepCCSocket& sock, IPC_ptr& ipc_sock) {
  auto info = sock.get_tcp_deepcc_info_json(
      DeepCCSocket::TCPInfoRequestType::REQUEST_ACTION);
  LOG(DEBUG) << "TCPDeepCCInfo" << info.dump();
  /*
   * write info to IPC socket
   * info should be string dumped from json
   */
  json state = info;
  uint16_t len = state.dump().length();
  ipc_sock->write(put_field(len) + state.dump());

  /* read feedback, feedback shall come from IPC socket */
  auto header = ipc_sock->read_exactly(2);
  auto action_len = get_uint16(header.data());
  auto action = ipc_sock->read_exactly(action_len);

  int cwnd = json::parse(action).at("cwnd");
  try {
    LOG(DEBUG) << "GET cwnd from user: " << cwnd;
    sock.set_tcp_cwnd(cwnd);
  } catch (const exception& e) {
    print_exception("set_cwnd", e);
    throw runtime_error("Cannot set tcp_cwnd: " + cwnd);
  }
}

void control_thread(DeepCCSocket& sock, IPC_ptr& ipc_sock,
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

void data_thread(TCPSocket& sock) {
  int file = open("/tmp/large-file.txt", O_RDONLY);
  char data[1024];
  while (send_traffic) {
    int len = read(file, (void*)&data, 1024);
    if (len > 0)
      sock.write(data);
    else
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
  std::chrono::milliseconds control_interval(10ms);
  if (not ipc_file.empty()) {
    ipc = std::move(setup_ipc_socket(ipc_file));
    control_interval = std::move(std::chrono::milliseconds(stoi(interval)));
    LOG(INFO) << "IPC with env has been established, control interval is "
              << control_interval.count() << "ms";
  }

  /* default CC is cubic */
  if (cong_ctl.empty()) {
    cong_ctl = "cubic";
  }

  int port = stoi(service);
  // init server addr
  Address address(ip, port);
  /* set reuse_addr */
  DeepCCSocket client;
  client.set_reuseaddr();
  client.set_congestion_control(cong_ctl);
  client.set_nodelay();
  LOG(INFO) << "Client set congestion control to " << cong_ctl;
  client.connect(address);
  client.enable_deepcc(2);
  LOG(INFO) << "Client enables deepCC plugin: 2";
  // LOG(INFO) << "Client connected to server";

  /* start data thread and control thread */
  thread ct;
  if (ipc != nullptr) {
    ct = std::move(thread(control_thread, std::ref(client), std::ref(ipc),
                          control_interval));
    LOG(INFO) << "Started control thread ... ";
  }
  thread dt(data_thread, std::ref(client));

  /* wait for finish */
  LOG(INFO) << "Client is sending data ... ";
  dt.join();
}
