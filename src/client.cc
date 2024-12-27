#include <getopt.h>
#include <signal.h>
#include <stdio.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "address.hh"
#include "common.hh"
#include "current_time.hh"
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
using IPC_ptr = std::unique_ptr<IPCSocket>;

Poller poller{};
// send_traffic should be atomic
std::atomic<bool> send_traffic(true);
std::atomic<bool> do_polling(true);
int global_flow_id = -1;
IPC_ptr ipc = nullptr;
std::chrono::_V2::system_clock::time_point ts_now = clock_type::now();

/* define message type */
enum class MessageType { INIT = 0, START = 1, END = 2, ALIVE = 3, OBSERVE = 4 };

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

void ipc_send_message(IPC_ptr& ipc_sock, const MessageType& type,
                      const json& state, const int observer_id = -1,
                      const int step = -1) {
  json message;
  message["state"] = state;
  message["flow_id"] = global_flow_id;
  if (type == MessageType::OBSERVE) {
    message["type"] = to_underlying(MessageType::OBSERVE);
    message["observer"] = observer_id;
    message["step"] = step;
  } else {
    // we just need to copy the type
    message["type"] = to_underlying(type);
  }

  uint16_t len = message.dump().length();
  if (ipc_sock) {
    ipc_sock->write(put_field(len) + message.dump());
  }
}

void signal_handler(int sig) {
  if (sig == SIGINT or sig == SIGKILL or sig == SIGTERM) {
    if (ipc) {
      /* inform IPC to close */
      ipc_send_message(ipc, MessageType::END, json({}));
      // LOG(INFO) << "Client " << global_flow_id << " has sent exiting
      // message";
    }
    LOG(INFO) << "Caught signal, Client " << global_flow_id << " exiting...";
    // first disable read from fd
    poller.remove_fd(ipc->fd_num());
    // disable write to IPC
    ipc->set_disconnected();
    do_polling = false;
    send_traffic = false;
    // IPC socket will be closed later
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    exit(1);
  }
}

std::unique_ptr<IPCSocket> setup_ipc_socket(const string& ipc_file) {
  if (fs::exists(ipc_file) && (not fs::is_socket(ipc_file)))
    throw runtime_error("IPC file exists but it is not a socket file");
  IPCSocket tmp_ipc;
  tmp_ipc.connect(ipc_file);
  // tmp_ipc.set_blocking(false);
  bool blocking = tmp_ipc.check_blocking();
  if (not blocking) {
    LOG(INFO) << "Client " << global_flow_id << " IPC set to non-blocking ";
  }
  LOG(INFO) << "Client " << global_flow_id << " IPC connected to " << ipc_file;
  // we need move semantics here to avoid using the deleted
  // copy constructor of FileDescriptor
  auto ipc = std::make_unique<IPCSocket>(std::move(tmp_ipc));
  LOG(INFO) << "Client " << global_flow_id
            << " IPCSocket fd: " << ipc->fd_num();
  // send current flow id to controller
  ipc_send_message(ipc, MessageType::INIT, json({}));
  // ipc->set_blocking(false);

  return ipc;
}

void do_congestion_control(DeepCCSocket& sock, IPC_ptr& ipc_sock) {
  auto data = sock.get_tcp_deepcc_info_json(RequestType::REQUEST_ACTION);
  LOG(TRACE) << "Client " << global_flow_id << " send state: " << data.dump();
  ipc_send_message(ipc_sock, MessageType::ALIVE, data);
  // set timestamp
  ts_now = clock_type::now();
  // action will be applied later
}

void do_poll() {
  while (do_polling.load()) {
    auto ret = poller.poll(-1);
    if (ret.result != Poller::Result::Type::Success) {
      exit(ret.exit_status);
    }
  }
}

void control_thread(DeepCCSocket& sock, IPC_ptr& ipc,
                    const std::chrono::milliseconds interval) {
  // register IPCSocket into poller
  poller.add_action(Poller::Action(
      *ipc, Direction::In,
      // callback
      [&]() -> ResultType {
        auto header = ipc->read_exactly(2);
        auto data_len = get_uint16(header.data());
        auto data = ipc->read_exactly(data_len);
        int type = json::parse(data).at("type");
        if (type == static_cast<int>(MessageType::OBSERVE)) {
          // observer wants to observe the world
          int observer = json::parse(data).at("observer");
          int step = json::parse(data).at("step");
          LOG(TRACE) << "Client " << global_flow_id
                     << " received message from observer: " << observer
                     << ", step: " << step << " to observe to world";
          auto data = sock.get_tcp_deepcc_info_json(RequestType::OBSERVE);
          ipc_send_message(ipc, MessageType::OBSERVE, data, observer, step);
        } else if (type == static_cast<int>(MessageType::ALIVE)) {
          // simple massage to enforce action
          int flow_id = json::parse(data).at("flow_id");
          int cwnd = json::parse(data).at("cwnd");
          sock.set_tcp_cwnd(cwnd);
          auto elapsed = clock_type::now() - ts_now;
          LOG(DEBUG) << "Client " << global_flow_id
                     << " GET cwnd from user: " << cwnd << ", elapsed time is "
                     << std::chrono::duration_cast<std::chrono::microseconds>(
                            elapsed)
                            .count()
                     << "us";
        }
        return ResultType::Continue;
      },
      // always interested
      [&]() { return true; },
      // err callback
      [&]() {
        LOG(ERROR) << "Client " << global_flow_id << " error on polling ";
        //  << ipc->fd_num() << ", error number is " << errno;
      }));

  // start polling thread
  thread polling_thread(do_poll);

  // start regular congestion control parttern
  auto when_started = clock_type::now();
  auto target_time = when_started + interval;
  while (send_traffic.load()) {
    do_congestion_control(sock, ipc);
    std::this_thread::sleep_until(target_time);
    target_time += interval;
  }
  polling_thread.join();
}

void data_thread(TCPSocket& sock) {
  string data(BUFSIZ, 'a');
  while (send_traffic.load()) {
    sock.write(data, true);
  }
  LOG(INFO) << "Data thread exits";
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
    case 'f':
      id = optarg;
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

  /* assign flow_id */
  if (not id.empty()) {
    global_flow_id = stoi(id);
    LOG(INFO) << "Flow id: " << global_flow_id;
  }

  std::chrono::milliseconds control_interval(10ms);
  if (not ipc_file.empty()) {
    // ipc = setup_ipc_socket(ipc_file);
    ipc = std::move(setup_ipc_socket(ipc_file));
    control_interval = std::move(std::chrono::milliseconds(stoi(interval)));
    LOG(INFO) << "Client " << global_flow_id
              << " IPC with env has been established, control interval is "
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
  LOG(DEBUG) << "Client " << global_flow_id << " set congestion control to "
             << cong_ctl;
  /* !! should be set after socket connected */
  int enable_deepcc = 2;
  client.enable_deepcc(enable_deepcc);
  LOG(DEBUG) << "Client " << global_flow_id << " "
             << "enables deepCC plugin: " << enable_deepcc;

  /* ignore SIGPIPE generated by Socket write */
  if (signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    throw runtime_error("signal: failed to ignore SIGPIPE");
  }

  /* start data thread and control thread */
  thread ct;
  if (ipc != nullptr) {
    ct = std::move(thread(control_thread, std::ref(client), std::ref(ipc),
                          control_interval));
    LOG(DEBUG) << "Client " << global_flow_id << " Started control thread ... ";
  }
  thread dt(data_thread, std::ref(client));
  LOG(INFO) << "Client " << global_flow_id << " is sending data ... ";

  /* wait for finish */
  dt.join();
  ct.join();
  // LOG(INFO) << "Joined data thread, to exiting ... sleep for a while";
}
