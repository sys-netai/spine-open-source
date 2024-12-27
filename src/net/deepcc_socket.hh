#ifndef DEEPCC_SOCKET_HH
#define DEEPCC_SOCKET_HH

#include <linux/tcp.h>
#include <sys/socket.h>

#include <mutex>
#include <queue>

#include "address.hh"
#include "exception.hh"
#include "file_descriptor.hh"
#include "socket.hh"
#include "tcp_info.hh"

using namespace std;

class DeepCCSocket : public TCPSocket {
 public:
  enum class TCPInfoRequestType : int { REQUEST_ACTION = 0, OBSERVE = 1 };

 protected:
  DeepCCSocket(FileDescriptor&& fd);

 public:
  DeepCCSocket();
  void enable_deepcc(int val);
  TCPDeepCCInfo get_tcp_deepcc_info(TCPInfoRequestType type);
  json get_tcp_deepcc_info_json(TCPInfoRequestType type);
  void set_tcp_cwnd(int cwnd);
  DeepCCSocket accept();
  /* get and set socket option */
  template <typename option_type>
  socklen_t getsockopt(const int level, const int option,
                       option_type& option_value) const;

  template <typename option_type>
  void setsockopt(const int level, const int option,
                  const option_type& option_value);

  /* get max throughput */
  uint64_t get_max_tput() const { return max_tput_; }

 private:
  void init();
  void prepare_request_info(TCPDeepCCInfo& info);
  void prepare_observe_info(TCPDeepCCInfo& dst, const TCPDeepCCInfo& src);

 private:
  bool tcp_deepcc_enable;
  std::queue<TCPDeepCCInfo> queue_{};
  /* maximal observed throughput */
  uint64_t max_tput_;
  /* last observed time in us */
  uint64_t last_observe_ts_;
  /* last request time in us */
  uint64_t last_request_ts_;
  /* last TCP information for request CWND */
  TCPDeepCCInfo last_request_info_;
  /* last TCP information for observer */
  TCPDeepCCInfo last_observe_info_;
  /* has observe between two request or not */
  bool has_observe_;
  /* mutex for avoiding concurrent get info */
  std::mutex mutex_;
};

#endif  // DEEPCC_SOCKET_HH