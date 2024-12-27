#ifndef TCP_INFO_HH
#define TCP_INFO_HH

#include <linux/tcp.h>
#include <sys/types.h>

#include <sstream>
#include <string>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

typedef unsigned int u32;
typedef int s32;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint8_t u8;

/**
 * @brief DeepCC infomation structure of kernel TCP
 * * NOTE that avg_urtt, cnt, avg_thr, thr_cnt, loss_bytes are associated with
 * * monitor interval
 *
 * If multiple observation operations are conducted within one Monitor interval,
 * intermediate results need to be preserved and merged with final observation
 * of this MI.
 *
 */
struct TCPDeepCCInfo {
  u32 min_rtt;  /* min-filtered RTT in us */
  u32 avg_urtt; /* averaged RTT in us from the previous info request till now */
  u32 cnt;      /* number of RTT samples used for averaging */
  u64 avg_thr;  /* Bytes per second */
  u32 thr_cnt;  /* Number of sampled throughput for averaging it */
  u32 cwnd;     /* congestion window */
  u32 pacing_rate;  /* pacing rate */
  u32 lost_bytes;   /* number of losses bytes (count from last monitor action to
                       now) */
  u32 srtt_us;      /* smoothed round trip time << 3 in usecs */
  u32 snd_ssthresh; /* Slow start size threshold */
  u32 packets_out;  /* Packets which are "in flight" */
  u32 retrans_out;  /* Retransmitted packets out */
  u32 max_packets_out; /* max packets_out in last window */
  u32 mss; /* mss size; throughput = rate->delivered * sk->mss_cache /
              rate->interval_us */

  void init() {
    min_rtt = 0;
    avg_urtt = 0;
    cnt = 0;
    avg_thr = 0;
    thr_cnt = 0;
    cwnd = 0;
    pacing_rate = 0;
    lost_bytes = 0;
    srtt_us = 0;
    snd_ssthresh = 0;
    packets_out = 0;
    retrans_out = 0;
    max_packets_out = 0;
    mss = 0;
  }
  TCPDeepCCInfo& operator=(const TCPDeepCCInfo& a) {
    this->min_rtt = a.min_rtt;
    this->avg_urtt = a.avg_urtt;
    this->cnt = a.cnt;
    this->avg_thr = a.avg_thr;
    this->thr_cnt = a.thr_cnt;
    this->cwnd = a.cwnd;
    this->pacing_rate = a.pacing_rate;
    this->lost_bytes = a.lost_bytes;
    this->srtt_us = a.srtt_us;
    this->snd_ssthresh = a.snd_ssthresh;
    this->packets_out = a.packets_out;
    this->retrans_out = a.retrans_out;
    this->max_packets_out = a.max_packets_out;
    this->mss = a.mss;
  }

  json to_json() {
    json out;
    out["min_rtt"] = min_rtt;
    out["avg_urtt"] = avg_urtt;
    out["cnt"] = cnt;
    out["cwnd"] = cwnd;
    out["avg_thr"] = avg_thr;
    out["thr_cnt"] = thr_cnt;
    out["pacing_rate"] = pacing_rate;
    out["loss_bytes"] = lost_bytes;
    out["srtt_us"] = srtt_us;
    out["snd_ssthresh"] = snd_ssthresh;
    out["retrans_out"] = retrans_out;
    out["packets_out"] = packets_out;
    out["max_packets_out"] = max_packets_out;
    out["mss_cache"] = mss;
    return out;
  }

  std::string to_string() {
    json out = this->to_json();
    return out.dump();
  }

  /**
   * @brief Merge MI-associated variables from src to des
   *
   * @param des destination info, maybe final observation of one Monitor
   * Interval
   * @param src source info, maybe intermediate observations
   */
  void merge_info(const TCPDeepCCInfo& src) {
    u32 total_rtt_cnt = this->cnt + src.cnt;
    u32 total_thr_cnt = this->thr_cnt + src.thr_cnt;

    // cnt and thr_cnt may be zero

    u32 avg_rtt_us = (this->cnt * this->avg_urtt + src.cnt * src.avg_urtt) /
                     std::max(total_rtt_cnt, u32(1));
    u32 avg_tput_bps =
        (this->thr_cnt * this->avg_thr + src.thr_cnt * src.avg_thr) /
        std::max(total_thr_cnt, u32(1));

    // merge results into des
    this->thr_cnt = total_thr_cnt;
    this->cnt = total_rtt_cnt;
    this->avg_thr = avg_tput_bps;
    this->avg_urtt = avg_rtt_us;
    this->lost_bytes += src.lost_bytes;
  }
};

#endif  // TCP_INFO_HH