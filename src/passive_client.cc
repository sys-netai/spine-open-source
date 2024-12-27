#include <getopt.h>
#include <signal.h>
#include <stdio.h>

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
#include "json.hpp"
#include "logging.hh"
#include "poller.hh"
#include "serialization.hh"
#include "socket.hh"
#include "tcp_info.hh"

using namespace std;
using namespace std::literals;
using clock_type = std::chrono::high_resolution_clock;
using namespace PollerShortNames;
typedef DeepCCSocket::TCPInfoRequestType RequestType;

// short name
using json = nlohmann::json;
using IPC_ptr = std::unique_ptr<FileDescriptor>;
using Callback = std::function<void(const int)>;

Poller poller{};
bool send_traffic = true;
int flow_id = -1;
IPC_ptr ipc = nullptr;

void signal_handler(int sig) {
  if (sig == SIGINT or sig == SIGKILL or sig == SIGTERM) {
    send_traffic = false;
    if (ipc != nullptr) {
      /* inform IPC to close */
      json message;
      message["state"] = "";
      message["tun_id"] = flow_id;
      message["end"] = 1;
      uint16_t len = message.dump().length();
      ipc->write(put_field(len) + message.dump());
    }
    LOG(INFO) << "Caught signal, exiting...";
    exit(1);
  }
}

std::unique_ptr<FileDescriptor> setup_ipc_socket(const string& ipc_file) {
  if (fs::exists(ipc_file) && (not fs::is_socket(ipc_file)))
    throw runtime_error("IPC file exists but it is not a socket file");
  IPCSocket tmp_ipc;
  tmp_ipc.connect(ipc_file);
  LOG(INFO) << "Client IPC connected to " << ipc_file;
  // send current flow id to controller
  json msg;
  msg["tun_id"] = flow_id;
  msg["state"] = "";
  msg["id"] = 1;
  uint16_t len = msg.dump().length();
  tmp_ipc.write(put_field(len) + msg.dump());
  // we need move semantics here to avoid using the deleted copy constructor of
  // FileDescriptor
  return std::make_unique<FileDescriptor>(std::move(tmp_ipc));
}

void control_thread(DeepCCSocket& client, IPC_ptr& ipc,
                    std::chrono::milliseconds interval = 10ms, int id = -1) {
  poller.add_action(Poller::Action(
      *ipc, Direction::In,
      // callback
      [&]() -> ResultType {
        auto header = ipc->read_exactly(2);
        auto reply_len = get_uint16(header.data());
        auto reply = ipc->read_exactly(reply_len);
        int cwnd = -1, flow = -1, msg = -1;
        auto data = json::parse(reply);
        try {
          cwnd = data.at("cwnd");
          flow = data.at("tun_id");
          msg = data.at("msg");
          LOG(DEBUG) << "GET cwnd for flow " << flow << " from user: " << cwnd
                     << "; msg is " << msg;
        } catch (const exception& e) {
          print_exception("set_cwnd", e);
          throw runtime_error("Cannot get control message from user");
        }
        // wait for cwnd enforcement or not
        if (msg == 1) {
          // enforce cwnd and wait for effect
          client.set_tcp_cwnd(cwnd);
          // prepare state
          auto target_time = clock_type::now() + interval;
          std::this_thread::sleep_until(target_time);
        }
        json info =
            client.get_tcp_deepcc_info_json(RequestType::REQUEST_ACTION);
        LOG(TRACE) << info.dump();
        /*
         * write info to IPC socket
         * info should be string dumped from json
         */
        json state = info;
        json message;
        message["state"] = state;
        message["tun_id"] = flow;
        uint16_t len = message.dump().length();
        ipc->write(put_field(len) + message.dump());
      },
      // when interested
      []() { return true; },
      // err callback
      [&]() { LOG(ERROR) << "error on polling " << ipc->fd_num(); }));
  while (send_traffic) {
    auto ret = poller.poll(-1);
    if (ret.result != Poller::Result::Type::Success) {
      exit(ret.exit_status);
    }
  }
}

void data_thread(TCPSocket& sock) {
  string data('a', BUFSIZ);
  while (send_traffic) {
    sock.write(data);
  }
}

void usage_error(const string& program_name) {
  cerr << "Usage: " << program_name << " [OPTION]... [COMMAND]" << endl;
  cerr << endl;
  cerr << "Options = --ip=IP_ADDR --port=PORT --cong=ALGORITHM --ipc=IPC_FILE "
          "--interval=INTERVAL (Milliseconds) --id=None"
       << endl;
  cerr << endl;
  cerr << "Default congestion control algorithms for incoming TCP is CUBIC; "
       << "Default control interval is 10ms; "
       << "Default flow id is None" << endl;

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
      {"id", optional_argument, nullptr, 'f'},
      {0, 0, nullptr, 0}};

  string ip, service, cong_ctl, ipc_file, interval, id;
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
    case 'f':
      id = optarg;
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

  /* assign flow_id */
  if (not id.empty()) {
    flow_id = stoi(id);
    LOG(INFO) << "Flow id: " << flow_id;
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
  client.connect(address);

  client.set_congestion_control(cong_ctl);
  client.set_nodelay();
  LOG(DEBUG) << "Client set congestion control to " << cong_ctl;
  /* !! should be set after socket connected */
  int enable_deepcc = 2;
  client.enable_deepcc(enable_deepcc);
  LOG(DEBUG) << "Client enables deepCC plugin: " << enable_deepcc;

  /* start data thread and control thread */
  thread ct;
  if (ipc != nullptr) {
    // add ipc to poller
    ct = std::move(thread(control_thread, std::ref(client), std::ref(ipc),
                          control_interval, flow_id));
    LOG(DEBUG) << "Started control thread ... ";
  }
  thread dt(data_thread, std::ref(client));

  /* wait for finish */
  LOG(INFO) << "Client is sending data ... ";
  dt.join();
}
