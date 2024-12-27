#ifndef IPC_SOCKET_HH
#define IPC_SOCKET_HH

#include <sys/socket.h>

#include <atomic>
#include <string>

#include "file_descriptor.hh"

/* UNIX domain socket */
class IPCSocket : public FileDescriptor {
 public:
  IPCSocket();

  IPCSocket(const IPCSocket& ipc) = delete;
  const IPCSocket& operator=(const FileDescriptor& other) = delete;

  /* construct from file descriptor, used by accept */
  IPCSocket(FileDescriptor&& s_fd, const int domain, const int type);

  IPCSocket(IPCSocket&& other);

  void bind(const std::string& path);
  void connect(const std::string& path);

  void listen(const int backlog = 20);

  bool check_blocking();

  IPCSocket accept(void);

  void set_reuseaddr(void);

  /* This function will tag IPC as disconnected, IPC socket will be closed in
   * destrctor of FileDescriptor */
  inline void set_disconnected(void) { connected_.store(false); }

  /* override write; add sanity check*/
  virtual std::string::const_iterator write(const std::string& buffer,
                                            const bool write_all = true);

 protected:
  /* get and set socket option */
  template <typename option_type>
  void setsockopt(const int level, const int option, const option_type& value);

  template <typename option_type>
  socklen_t getsockopt(const int level, const int option,
                       option_type& option_value) const;

 private:
  std::atomic<bool> connected_;
};

#endif /* IPC_SOCKET_HH */