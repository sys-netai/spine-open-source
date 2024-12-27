#include <iostream>
#include <stdexcept>
#include <string>

#include "filesystem.hh"
#include "ipc_socket.hh"
#include "json.hpp"
#include "logging.hh"
#include "serialization.hh"

using json = nlohmann::json;
using IPC_ptr = std::unique_ptr<FileDescriptor>;
using namespace std;

std::unique_ptr<FileDescriptor> setup_ipc_socket(const string& ipc_file) {
  if (fs::exists(ipc_file) && not fs::is_socket(ipc_file))
    throw runtime_error("IPC file exists but it is not a socket file");
  IPCSocket ipc;
  ipc.bind(ipc_file);
  ipc.listen();
  LOG(INFO) << "Sender IPC listen at " << ipc_file;
  return std::make_unique<FileDescriptor>(ipc.accept());
}

int main(int argc, char** argv) {
  auto ipc_sock = setup_ipc_socket("/home/xudong/ipc_file");
  json data;
  data["name"] = "wxc";
  data["time"] = 1;
  uint16_t len = data.dump().length();
  ipc_sock->write(put_field(len) + data.dump());

  auto header = ipc_sock->read_exactly(2);
  auto action_len = get_uint16(header.data());
  auto action = ipc_sock->read_exactly(action_len);
  int cwnd = json::parse(action).at("cwnd");
  LOG(INFO) << "read from Socket: " << cwnd;

  return 0;
}