#ifndef TCP_SPINE_H
#define TCP_SPINE_H

#include "lib/spine.h"

#include <linux/kernel.h>
#include <linux/net.h>
#include <linux/tcp.h>
#include <net/tcp.h>

#define MAX_SKB_STORED 50

#define MAX_ACTIVE_FLOWS 1024
#define MAX_DATAPATH_PROGRAMS 10

struct skb_info {
	u64 first_tx_mstamp; // choose the correct skb so the timestamp for first packet
	u32 interval_us; // interval us as calculated from this SKB
};

struct spine {
	// control
	u32 last_snd_una; // 4 B
	u32 last_bytes_acked; // 8 B
	u32 last_sacked_out; // 12 B
	struct skb_info *skb_array; // array of future skb information
};

#define MTU 1500
#define S_TO_US 1000000

static inline void get_sock_from_spine(struct sock **sk,
				       struct spine_connection *conn)
{
	*sk = (struct sock *)spine_get_impl(conn);
}

void spine_set_pacing_rate(struct sock *sk, uint32_t rate);
void spine_log(struct spine_datapath *dp, enum spine_log_level level,
	       const char *msg, int msg_size);

u64 spine_now(void);
u64 spine_since(u64 then);
u64 spine_after(u64 us);

#endif
