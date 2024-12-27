#ifndef SPINE_H
#define SPINE_H

#ifdef __KERNEL__
#include <linux/module.h>
#include <linux/types.h>
#else
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef enum {
	SPINE_CUBIC = 0,
	SPINE_VEGAS,
	SPINE_VANILLA,
	SPINE_NEO
} spine_internal_alg;

enum spine_log_level {
	TRACE,
	DEBUG,
	INFO,
	WARN,
	ERROR,
};

struct spine_primitives {
	// newly acked, in-order bytes
	u32 bytes_acked;
	// newly acked, in-order packets
	u32 packets_acked;
	// out-of-order bytes
	u32 bytes_misordered;
	// out-of-order packets
	u32 packets_misordered;
	// bytes corresponding to ecn-marked packets
	u32 ecn_bytes;
	// ecn-marked packets
	u32 ecn_packets;

	// an estimate of the number of packets lost
	u32 lost_pkts_sample;
	// whether a timeout was observed
	bool was_timeout;

	// a recent sample of the round-trip time
	u64 rtt_sample_us;
	// sample of the sending rate, bytes / s
	u64 rate_outgoing;
	// sample of the receiving rate, bytes / s
	u64 rate_incoming;
	// the number of actual bytes in flight
	u32 bytes_in_flight;
	// the number of actual packets in flight
	u32 packets_in_flight;
	// the target congestion window to maintain, in bytes
	u32 snd_cwnd;
	// target rate to maintain, in bytes/s
	u64 snd_rate;

	// amount of data available to be sent
	// NOT per-packet - an absolute measurement
	u32 bytes_pending;
};

// maximum string length for congAlg
#define MAX_CONG_ALG_SIZE 64
/* Datapaths provide connection information to ccp_connection_start
 */
struct spine_datapath_info {
	u32 init_cwnd;
	u32 mss;
	u32 src_ip;
	u32 src_port;
	u32 dst_ip;
	u32 dst_port;
	spine_internal_alg alg;
	char congAlg[MAX_CONG_ALG_SIZE];
};

struct spine_connection {
	// the index of this array element
	u16 index;

	u64 last_create_msg_sent;
	// parameters of this tcp connection
	// u64 *parameters;
	// u8 num_params;

	// struct spine_primitives is large; as a result, we store it inside spine_connection to avoid
	// potential limitations in the datapath
	// datapath should update this before calling spine_invoke()
	// struct spine_primitives prims;

	// constant flow-level information
	struct spine_datapath_info flow_info;

	// private libspine state for the send machine and measurement machine
	void *state;

	// datapath-specific per-connection state
	void *impl;

	// pointer back to parent datapath that owns this connection
	struct spine_datapath *datapath;
};

struct spine_datapath {
	// control primitives
	void (*set_cwnd)(struct spine_connection *conn, u32 cwnd);
	void (*set_rate_abs)(struct spine_connection *conn, u32 rate);
	void (*set_params)(struct spine_connection *conn, u64 *params,
			   u8 num_fields);
	/* fetch measurements from cc module. if success, the num_fields is set with a positive value and feilds is stored in measurements */
	void (*fetch_measurements)(struct spine_connection *conn,
				   u64 *measurements, u8 *num_fields,
				   u32 request_index);

	// IPC communication
	int (*send_msg)(struct spine_datapath *dp, char *msg, int msg_size);

	// logging
	void (*log)(struct spine_datapath *dp, enum spine_log_level level,
		    const char *msg, int msg_size);

	// time management
	u64 time_zero;
	u64 (*now)(void); // the current time in datapath time units
	u64 (*since_usecs)(u64 then); // elapsed microseconds since <then>
	u64 (*after_usecs)(
		u64 usecs); // <usecs> microseconds from now in datapath time units
	size_t max_connections;
	// list of active connections this datapath is handling
	struct spine_connection *spine_active_connections;
	u64 fto_us;
	u64 last_msg_sent;
	bool _in_fallback;

	// datapath-specific global state, such as: for sock* sk
	void *impl;
};

int spine_init(struct spine_datapath *datapath, u32 id);

void spine_free(struct spine_datapath *datapath);

struct spine_connection *
spine_connection_start(struct spine_datapath *datapath, void *impl,
		       struct spine_datapath_info *flow_info);

struct spine_connection *
spine_connection_lookup(struct spine_datapath *datapath, u16 sid);

void spine_connection_free(struct spine_datapath *datapath, u16 sid);

// real underlying datapath implementation: linux kernel socket or quic or ...
void *spine_get_impl(struct spine_connection *conn);

void spine_set_impl(struct spine_connection *conn, void *ptr);

// communication
int spine_read_msg(struct spine_datapath *datapath, char *buf, int bufsize);

// the ultimate function called in congestion control logic
int spine_invoke(struct spine_connection *conn);

// timing
void _update_fto_timer(struct spine_datapath *datapath);
bool _check_fto(struct spine_datapath *datapath);
void _turn_off_fto_timer(struct spine_datapath *datapath);

#endif
