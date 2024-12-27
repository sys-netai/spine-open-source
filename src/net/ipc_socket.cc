#include "ipc_socket.hh"

#include <sys/fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "exception.hh"

using namespace std;

IPCSocket::IPCSocket()
    : FileDescriptor(SystemCall("socket", socket(AF_UNIX, SOCK_STREAM, 0))),
      connected_(false) {}

IPCSocket::IPCSocket(FileDescriptor&& fd, const int domain, const int type)
    : FileDescriptor(move(fd)), connected_(true) {
  int actual_value;
  socklen_t len;

  /* verify domain */
  len = getsockopt(SOL_SOCKET, SO_DOMAIN, actual_value);
  if ((len != sizeof(actual_value)) or (actual_value != domain)) {
    throw runtime_error("socket domain mismatch");
  }

  /* verify type */
  len = getsockopt(SOL_SOCKET, SO_TYPE, actual_value);
  if ((len != sizeof(actual_value)) or (actual_value != type)) {
    throw runtime_error("socket type mismatch");
  }
}

IPCSocket::IPCSocket(IPCSocket&& other)
    : FileDescriptor(reinterpret_cast<FileDescriptor&&>(other)) {
  if (other.fd_num() != -1)
    throw runtime_error("IPCSocket: move constructor failed");
  connected_.store(other.connected_.load());
}

sockaddr_un create_sockaddr_un(const string& path) {
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    throw runtime_error("path size is too long");
  }

  struct sockaddr_un addr;
  addr.sun_family = AF_UNIX;
  strcpy(addr.sun_path, path.c_str());
  return addr;
}

void IPCSocket::bind(const string& path) {
  const sockaddr_un addr = create_sockaddr_un(path);
  SystemCall("bind",
             ::bind(fd_num(), (struct sockaddr*)&addr, sizeof(sockaddr_un)));
}

void IPCSocket::connect(const string& path) {
  const sockaddr_un addr = create_sockaddr_un(path);
  SystemCall("connect",
             ::connect(fd_num(), (struct sockaddr*)&addr, sizeof(sockaddr_un)));
  connected_ = true;
}

void IPCSocket::listen(const int backlog) {
  SystemCall("listen", ::listen(fd_num(), backlog));
}

IPCSocket IPCSocket::accept() {
  register_read();
  return IPCSocket(SystemCall("accept", ::accept(fd_num(), nullptr, nullptr)),
                   AF_UNIX, SOCK_STREAM);
}

/* set socket option */
template <typename option_type>
void IPCSocket::setsockopt(const int level, const int option,
                           const option_type& option_value) {
  SystemCall("setsockopt", ::setsockopt(fd_num(), level, option, &option_value,
                                        sizeof(option_value)));
}

/* get socket option */
template <typename option_type>
socklen_t IPCSocket::getsockopt(const int level, const int option,
                                option_type& option_value) const {
  socklen_t optlen = sizeof(option_value);
  SystemCall("getsockopt",
             ::getsockopt(fd_num(), level, option, &option_value, &optlen));
  return optlen;
}

/* allow local address to be reused sooner, at the cost of some robustness */
void IPCSocket::set_reuseaddr(void) {
  setsockopt(SOL_SOCKET, SO_REUSEADDR, int(true));
}

bool IPCSocket::check_blocking() {
  int val = SystemCall("fcntl F_GETFL", fcntl(fd_num(), F_GETFL, 0));
  return !(val & O_NONBLOCK);
}

string::const_iterator IPCSocket::write(const std::string& buffer,
                                        const bool write_all) {
  if (not connected_.load()) return buffer.begin();

  auto it = buffer.begin();
  do {
    ssize_t bytes_written = ::write(fd_num(), &*it, buffer.end() - it);
    if (bytes_written <= 0 or errno == EPIPE or errno == EBADF) {
      connected_.store(false);
      return it;
    }
    register_write();
    it += bytes_written;
  } while (write_all and (it != buffer.end()));

  return it;
}