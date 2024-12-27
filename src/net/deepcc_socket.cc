#include "deepcc_socket.hh"

#include "common.hh"
#include "logging.hh"
#include "timestamp.hh"

#define SECOND_TO_US 1000000

using json = nlohmann::json;

DeepCCSocket::DeepCCSocket() : TCPSocket() { init(); }

DeepCCSocket::DeepCCSocket(FileDescriptor&& fd) : TCPSocket(std::move(fd)) {
  init();
}

void DeepCCSocket::init() {
  tcp_deepcc_enable = true;
  max_tput_ = 0;
  last_observe_ts_ = 0;
  last_request_ts_ = 0;
  last_observe_info_.init();
  last_request_info_.init();
  has_observe_ = false;

  // init timestamp
  initial_timestamp();
}

/* accept a new incoming connection */
DeepCCSocket DeepCCSocket::accept(void) {
  register_read();
  return DeepCCSocket(FileDescriptor(
      SystemCall("accept", ::accept(fd_num(), nullptr, nullptr))));
}

void DeepCCSocket::enable_deepcc(int val) {
  setsockopt(IPPROTO_TCP, TCP_DEEPCC_ENABLE, val);
  tcp_deepcc_enable = true;
}

TCPDeepCCInfo DeepCCSocket::get_tcp_deepcc_info(TCPInfoRequestType type) {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (not tcp_deepcc_enable) {
    throw runtime_error("DeepCC hasn't been activated");
  }
  struct TCPDeepCCInfo info;
  getsockopt(IPPROTO_TCP, TCP_DEEPCC_INFO, info);
  // record max throughput
  if (info.avg_thr > max_tput_) {max_tput_ = info.avg_thr;}
  else {max_tput_ = (uint64_t)(0.99 * max_tput_ + 0.01 * info.avg_thr);} //std::max(max_tput_, info.avg_thr);
  switch (type) {
  case TCPInfoRequestType::REQUEST_ACTION:
    LOG(TRACE) << "Empty queue, queue size is " << queue_.size();
    prepare_request_info(info);
    last_request_info_ = info;
    has_observe_ = false;
    break;

  case TCPInfoRequestType::OBSERVE:
    LOG(TRACE) << "Intermediate observation, push to queue and return";
    // first enqueue temp observation for preparing next Request
    queue_.emplace(info);
    // merge current observed info with last observed info
    const auto& last_observed =
        has_observe_ ? last_observe_info_ : last_request_info_;
    prepare_observe_info(info, last_observed);
    has_observe_ = true;
    last_observe_info_ = info;
  }
  return info;
}

json DeepCCSocket::get_tcp_deepcc_info_json(TCPInfoRequestType type) {
  uint64_t time_delta = 0;
  auto now = timestamp_usecs();
  switch (type) {
  case TCPInfoRequestType::REQUEST_ACTION:
    time_delta = now - last_request_ts_;
    last_request_ts_ = now;
    break;

  case TCPInfoRequestType::OBSERVE:
    time_delta = now - last_observe_ts_;
    last_observe_ts_ = now;
    break;
  }
  // timedelta in us
  time_delta = std::max(time_delta, u64(1));
  auto info = get_tcp_deepcc_info(type);
  // loss ratio in bytes per second
  auto loss_ratio = double(info.lost_bytes * SECOND_TO_US) / time_delta;
  auto data = std::move(info.to_json());
  // we also want to know the observed max throughput
  data["max_tput"] = max_tput_;
  data["loss_ratio"] = loss_ratio;
  data["time_delta"] = time_delta;
  return data;
}

void DeepCCSocket::prepare_request_info(TCPDeepCCInfo& info) {
  if (queue_.size() == 0) return;
  while (not queue_.empty()) {
    auto inter_observation = queue_.front();
    info.merge_info(inter_observation);
    queue_.pop();
  }
}

inline void DeepCCSocket::prepare_observe_info(TCPDeepCCInfo& dst,
                                               const TCPDeepCCInfo& src) {
  dst.merge_info(src);
}

void DeepCCSocket::set_tcp_cwnd(int cwnd) {
  if (not tcp_deepcc_enable) {
    throw runtime_error("DeepCC hasn't been activated");
  }
  setsockopt(IPPROTO_TCP, TCP_CWND, cwnd);
}

/* get socket option */
template <typename option_type>
socklen_t DeepCCSocket::getsockopt(const int level, const int option,
                                   option_type& option_value) const {
  socklen_t optlen = sizeof(option_value);
  SystemCall("getsockopt",
             ::getsockopt(fd_num(), level, option, &option_value, &optlen));
  return optlen;
}

/* set socket option */
template <typename option_type>
void DeepCCSocket::setsockopt(const int level, const int option,
                              const option_type& option_value) {
  SystemCall("setsockopt", ::setsockopt(fd_num(), level, option, &option_value,
                                        sizeof(option_value)));
}