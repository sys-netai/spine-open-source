#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "address.hh"
#include "common.hh"
#include "logging.hh"
#include "socket.hh"
using namespace std;

#define BUFFER 1024

void usage_error(const string& program_name) {
  cerr << "Usage: " << program_name << " [OPTION]... [COMMAND]" << endl;
  cerr << endl;
  cerr << "Options = --ip=IP_ADDR --port=PORT --cong=ALGORITHM (default: "
          "CUBIC)"
       << endl;
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
      {0, 0, nullptr, 0}};

  string service, cong_ctl;
  while (true) {
    const int opt = getopt_long(argc, argv, "", command_line_options, nullptr);
    if (opt == -1) { /* end of options */
      break;
    }
    switch (opt) {
    case 'c':
      cong_ctl = optarg;
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
  client.set_nodelay();
  LOG(DEBUG) << "Congestion control algorithm: " << cong_ctl;

  int out = open("/tmp/outfile.txt", O_CREAT | O_RDWR, 0666);
  while (true) {
    auto data = client.read(BUFFER);
    LOG(TRACE) << "receiver get: " << data;
    if (not data.empty()) {
      write(out, data.c_str(), data.size());
    } else {
      LOG(INFO) << "empty data, exiting...";
      break;
    }
  }
}