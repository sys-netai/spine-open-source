#define pr_fmt(fmt) "[spine]: " fmt

#include <linux/math64.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <net/tcp.h>

#include "lib/spine.h"
#include "spine_nl.h"
#include "tcp_spine.h"

/* all parameters are devided by 1024 */
#define NEO_SCALE 1000
#define CWND_GAIN 1200
#define NEO_ACTION_INCREASE 1025
#define NEO_ACTION_DECREASE 976
// #define NEO_ACTION_INCREASE_MINOR 1010
// #define NEO_ACTION_DECREASE_MINOR 990
#define NEO_PARAM_NUM 1

#define NEO_IGNORE_PACKETS 1

#define NEO_INTERVALS 20
#define MONITOR_INTERVAL 30000
#define NEO_RATE_MIN 4096u

extern struct spine_datapath *kernel_datapath;
extern struct timespec64 tzero;
static int id = 0;
struct neo_interval {
	u64 rate; /* sending rate of this interval, bytes/sec */

	u64 recv_start; /* timestamps for when interval was waiting for acks */
	u64 recv_end;

	u64 send_start; /* timestamps for when interval data was being sent */
	u64 send_end;

	u64 start_rtt; /* smoothed RTT at start and end of this interval */
	u64 end_rtt;

	u32 packets_sent_base; /* packets sent when this interval started */
	u32 packets_ended; /* packets sent when this interval ended */

	u32 lost; /* packets sent during this interval that were lost */
	u32 delivered; /* packets sent during this interval that were delivered */
};

/* TCP NEO Parameters */
struct neo_data {
	int cnt; /*  cwnd change */
	bool in_recovery;
	u32 r_cwnd; /* cwnd in loss or recovery */

	u8 slow_start_passed;

	/* neo parameters */
	struct neo_interval *intervals; /* containts stats for 1 RTT */

	int send_index; /* index of interval currently being sent */
	int receive_index; /* index of interval currently receiving acks */

	u64 rate; /* current sending rate */
	u64 ready_rate; /* rate updated by RL model, used in the next MI */
	// u64 last_used_cwnd; /* last used rate to inform the agent */

	u32 lost_base; /* previously lost packets */
	u32 delivered_base; /* previously delivered packets */

	u32 packets_counted; /* packets received or loss confirmed*/

	/* CA state on previous ACK */
	u32 prev_ca_state : 3;
	/* prior cwnd upon entering loss recovery */
	u32 prior_cwnd;

	bool first_circle;

	int id;
	/* communication */
	struct spine_connection *conn;

	/* others */
	u32 double_counted;
};

/*****************
 * Util functions *
 * ************/

static u32 get_next_index(u32 index)
{
	if (index < NEO_INTERVALS - 1)
		return index + 1;
	return 0;
}

static u32 get_previous_index(u32 index)
{
	if (index > 0)
		return index - 1;
	return NEO_INTERVALS - 1;
}

/*********************
 * Getters / Setters *
 * ******************/
static u32 neo_get_rtt(struct tcp_sock *tp)
{
	/* Get initial RTT - as measured by SYN -> SYN-ACK.
	 * If information does not exist - use 1ms as a "LAN RTT".
	 * (originally from BBR).
	 */
	if (tp->srtt_us) {
		return max(tp->srtt_us >> 3, 1U);
	} else {
		return USEC_PER_MSEC;
	}
}

/**
 * With the ready_cwnd given by the RL agent. Calculate the real cwnd so that the average CWND/rate of all the unreceived MIs is the ready_cwnd.
 * rate1+rate2+rate3+new_rate = ready_cwnd * n
 * Used after send_index++ (new interval created.)
 * */

void neo_calculate_and_set_rate(struct sock *sk, struct neo_data *neo,
				struct neo_interval *interval)
{
	struct tcp_sock *tp = tcp_sk(sk);
	u64 new_rate;
	u64 cwnd_sum = 0;
	int recv_idx = neo->receive_index;
	int send_idx = neo->send_index;
	int idx = recv_idx;
	int num = 0;
	// if (recv_idx != send_idx){
	// 	do {
	// 		rate_sum += neo->intervals[idx].rate;
	// 		idx = get_next_index(idx);
	// 		pr_info("add idx %d-th rate %llu, current rate sum: %llu, num: %d", idx, neo->intervals[idx].rate, rate_sum, num);
	// 		num++;
	// 	} while (idx != send_idx);
	// }
	new_rate = neo->ready_rate; // * (num + 1) - rate_sum;
	new_rate = max(new_rate, NEO_RATE_MIN);
	new_rate = min(new_rate, sk->sk_max_pacing_rate);
	interval->rate = new_rate;
	neo->rate = new_rate;
	neo->ready_rate = new_rate; // in case no action is given. reuse the previous cwnd.
	sk->sk_pacing_rate = new_rate;
	// pr_info("The ready rate is %llu, and the new rate is thus %llu, the ratio is %d", neo->ready_cwnd, new_rate, new_rate/neo->ready_cwnd);
	
}


static void neo_update_pacing_rate(struct sock *sk)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *neo = inet_csk_ca(sk);

	u64 rate;
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);

	rate = tcp_mss_to_mtu(sk, tcp_sk(sk)->mss_cache); //

	rate *= USEC_PER_SEC;

	rate *= max(tp->snd_cwnd, tp->packets_out);

	rate = rate >> 1;

	if (likely(tp->srtt_us >> 3))
		do_div(rate, tp->srtt_us >> 3);

	/* WRITE_ONCE() is needed because sch_fq fetches sk_pacing_rate
   * without any lock. We want to make sure compiler wont store
   * intermediate values in this location.
   */
	// pr_info("rate:%uul", rate);
	WRITE_ONCE(sk->sk_pacing_rate,
		   min_t(u64, rate, sk->sk_max_pacing_rate));
}

static void neo_set_cwnd(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	u64 cwnd = sk->sk_pacing_rate;
	u32 rtt = neo_get_rtt(tcp_sk(sk));
	cwnd *= rtt;
	cwnd /= tp->mss_cache;

	cwnd /= USEC_PER_SEC;
	cwnd *= CWND_GAIN;
	cwnd /= NEO_SCALE;
	cwnd = max(4ULL, cwnd);
	cwnd = min((u32)cwnd, tp->snd_cwnd_clamp); /* apply cap */
	tp->snd_cwnd = cwnd;
}

bool neo_valid(struct neo_data *neo)
{
	return (neo && neo->intervals);
}

/* Set the pacing rate and cwnd base on the currently-sending interval */
void start_interval(struct sock *sk, struct neo_data *neo)
{

	struct neo_interval *interval = &neo->intervals[neo->send_index];
	interval->packets_ended = 0;
	interval->lost = 0;
	interval->delivered = 0;
	interval->packets_sent_base = max(tcp_sk(sk)->data_segs_out, 1U);
	interval->send_start = tcp_sk(sk)->tcp_mstamp;
	// pr_info("Start a interval, packets_sent_base: %d, send_start:%llu\n", interval->packets_sent_base, interval->send_start);
	neo_calculate_and_set_rate(sk, neo, interval);
	neo_set_cwnd(sk);
}

/**************************
 * intervals & sample:
 * was started, was ended,
 * find interval per sample
 * ************************/

/* Have we sent all the data we need to for this interval? Must have at least a MONITER_INTERVAL.*/
bool send_interval_ended(struct neo_interval *interval, struct tcp_sock *tsk,
			 struct neo_data *neo)
{
	u64 now = tsk->tcp_mstamp;
	if (now - interval->send_start >= MONITOR_INTERVAL) {
		interval->packets_ended = tsk->data_segs_out;
		return true;
	} else
		return false;
}

/* Have we accounted for (acked or lost) enough of the packets that we sent to
 * calculate summary statistics?
 */
bool receive_interval_ended(struct neo_interval *interval, struct tcp_sock *tsk,
			    struct neo_data *neo)
{
	return interval->packets_ended &&
	       interval->packets_ended - NEO_IGNORE_PACKETS < neo->packets_counted;
}

/* Start the next interval's sending stage.
 */
void start_next_send_interval(struct sock *sk, struct neo_data *neo)
{
	neo->send_index = get_next_index(neo->send_index);
	if (neo->send_index == neo->receive_index) {
		printk(KERN_INFO "Fail: not enough interval slots.\n");
		return;
	}
	start_interval(sk, neo);
}

/* Update the receiving time window and the number of packets lost/delivered
 * based on socket statistics.
 */
void neo_update_interval(struct neo_interval *interval, struct neo_data *neo,
			 struct sock *sk)
{
	interval->recv_end = tcp_sk(sk)->tcp_mstamp;
	interval->end_rtt = tcp_sk(sk)->srtt_us >> 3;
	interval->lost += tcp_sk(sk)->lost - neo->lost_base;
	interval->delivered += tcp_sk(sk)->delivered - neo->delivered_base;
}

/* Updates the NEO model */
void neo_process(struct sock *sk)
{
	struct neo_data *neo = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);
	struct neo_interval *interval;
	int index;
	u32 before;

	if (!neo_valid(neo))
		return;
	// neo_update_pacing_rate(sk);
	/* update send intervals */
	interval = &neo->intervals[neo->send_index];
	if (send_interval_ended(interval, tsk, neo)) {
		// pr_info("sending inverval ended, start the next send at time %llu.", tsk->tcp_mstamp);
		interval->send_end = tcp_sk(sk)->tcp_mstamp;
		start_next_send_interval(sk, neo);
	}
	/* update recv intervals */
	index = neo->receive_index;
	interval = &neo->intervals[index];
	before = neo->packets_counted;
	neo->packets_counted = tsk->delivered + tsk->lost -
				neo->double_counted;
	if (before > NEO_IGNORE_PACKETS + interval->packets_sent_base) {
		// pr_info("update %d-th recv inverval.", index);
		neo_update_interval(interval, neo, sk);
	}
	if (receive_interval_ended(interval, tsk, neo)) {
		// pr_info("recving inverval ended id: %d, start the next receive at time %llu.", index, tsk->tcp_mstamp);
		neo->receive_index = get_next_index(index);
		interval = &neo->intervals[neo->receive_index];
		interval->recv_start = tcp_sk(sk)->tcp_mstamp;
		interval->start_rtt = tcp_sk(sk)->srtt_us >> 3;
		if (neo->receive_index == 0)
			neo->first_circle = false;
	}
}

/** 
 * Spine call this to push updated parameters.
 * The state features we need:
 *    rate: for the RL agent to calculate the next rate.
 *    thr_gradient: (thr_t - thr_{t-1})/thr_{t-1}
 *    rtt_gradient: (RTT_t - RTT_{t-1})/MI
 *    loss_gradient: (1-loss...)
 *    rate_gradient: rate_t/rate_{t-1}
 * 
 * The state the kernel can provide as integers:
 *     delivered, last_delivered, lost, last_loss, rate, last_rate, RTT diff, 
 *
 * ps: For now request_index is not used, just fetch the lastest MI.
 */

void neo_fetch_measurements(struct spine_connection *conn,
				   u64 *measurements, u8 *num_fields,
				   u32 request_index)
{
	struct sock *sk;
	get_sock_from_spine(&sk, conn);
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *neo = inet_csk_ca(sk);
	*num_fields = 10;
	if (neo->first_circle && neo->receive_index < 2) {
		measurements[0] = 0;
		measurements[1] = 0;
		measurements[2] = 0;
		measurements[3] = 0;
		measurements[4] = 0;
		measurements[5] = 0;
		measurements[6] = 0;
		measurements[7] = 0;
		measurements[8] = 0;
		measurements[9] = 0;
		return;
	}
	int last_received_id = get_previous_index(neo->receive_index);
	int last_last_received_id = get_previous_index(last_received_id);
	// neo->last_used_cwnd = neo->intervals[last_received_id].cwnd;

	// pr_info("For the last interval: rate: %llu, lost: %llu; delivered: %llu; start_Rtt:%llu, end_rtt:%llu. send_start:%llu, send_end:%llu, recv_start:%llu, recv_end:%llu ", 
	// 				neo->intervals[last_received_id].rate,
	// 				neo->intervals[last_received_id].lost,
	// 				neo->intervals[last_received_id].delivered,
	// 				neo->intervals[last_received_id].start_rtt,
	// 				neo->intervals[last_received_id].end_rtt,
	// 				neo->intervals[last_received_id].send_start,
	// 				neo->intervals[last_received_id].send_end,
	// 				neo->intervals[last_received_id].recv_start,
	// 				neo->intervals[last_received_id].recv_end);
	measurements[0] = neo->intervals[last_received_id].delivered;
	measurements[1] = neo->intervals[last_last_received_id].delivered;
	measurements[2] = neo->intervals[last_received_id].lost;
	measurements[3] = neo->intervals[last_last_received_id].lost;
	measurements[4] = neo->intervals[last_received_id].packets_ended - neo->intervals[last_received_id].packets_sent_base;
	measurements[5] = neo->intervals[last_last_received_id].packets_ended -  neo->intervals[last_last_received_id].packets_sent_base; 
	measurements[6] = neo->intervals[last_received_id].end_rtt;
	measurements[7]	= neo->intervals[last_received_id].start_rtt;
	measurements[8] = neo->intervals[last_received_id].recv_end -
		    neo->intervals[last_received_id].recv_start;	
	measurements[9] = neo->intervals[last_last_received_id].recv_end-
		    neo->intervals[last_last_received_id].recv_start;	
}

/**
 * Spine call this to fetch updated parameters.
 */
void neo_set_params(struct spine_connection *conn, u64 *params, u8 num_fields)
{
	struct sock *sk;
	// struct tcp_sock *tp = tcp_sk(sk);
	get_sock_from_spine(&sk, conn);
	struct neo_data *ca = inet_csk_ca(sk);

	if (conn == NULL || params == NULL) {
		pr_info("%s:conn/params is NULL\n", __FUNCTION__);
		return;
	}

	if (unlikely(conn->flow_info.alg != SPINE_NEO) ||
	    unlikely(num_fields != NEO_PARAM_NUM)) {
		pr_info("Unknown internal congestion control algorithm, do nothing. %d",
			num_fields);
		return;
	}
	if (params[0] == 1){
		ca->ready_rate = ca->rate * NEO_ACTION_INCREASE / NEO_SCALE + 1;
	}else if (params[0] == 2){
		ca->ready_rate = ca->rate * NEO_ACTION_DECREASE / NEO_SCALE - 1;
	// }else if (params[0] == 3){
	// 	ca->ready_rate = ca->rate * NEO_ACTION_INCREASE_MINOR / NEO_SCALE + 1;
	// }else if (params[0] == 4){
	// 	ca->ready_rate = ca->rate * NEO_ACTION_DECREASE_MINOR / NEO_SCALE - 1;
	}else{ // 0
		ca->ready_rate = ca->rate;
	}
}


static void neo_release(struct sock *sk)
{
	struct neo_data *ca = inet_csk_ca(sk);
	if (ca->conn != NULL) {
		pr_info("freeing connection %d", ca->conn->index);
		spine_connection_free(kernel_datapath, ca->conn->index);
	} else {
		pr_info("already freed");
	}
	id--;
	kfree(ca->intervals);
}

static inline void neo_reset(struct neo_data *ca)
{
	ca->cnt = 0;
	ca->prev_ca_state = TCP_CA_Open;
	ca->in_recovery = false;
	ca->prior_cwnd = 0;
	ca->r_cwnd = 0;
	ca->slow_start_passed = 0;
}

static void neo_init(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);
	neo_reset(ca);

	ca->intervals = kzalloc(sizeof(struct neo_interval) * NEO_INTERVALS,
				GFP_KERNEL);
	if (!ca->intervals) {
		printk(KERN_INFO "init fails\n");
		return;
	}

	id++;
	ca->id = id;
	ca->rate = NEO_RATE_MIN * 512;
	ca->ready_rate = NEO_RATE_MIN * 512;
	// ca->last_used_cwnd = 10U;

	ca->send_index = 0;
	ca->receive_index = 0;
	ca->first_circle = true;
	ca->double_counted = 0;

	start_interval(sk, ca);

	/* create spine flow and register parameters */
	struct spine_datapath_info dp_info = {
		.init_cwnd = tp->snd_cwnd * tp->mss_cache,
		.mss = tp->mss_cache,
		.src_ip = tp->inet_conn.icsk_inet.inet_saddr,
		.src_port = tp->inet_conn.icsk_inet.inet_sport,
		.dst_ip = tp->inet_conn.icsk_inet.inet_daddr,
		.dst_port = tp->inet_conn.icsk_inet.inet_dport,
		.congAlg = "neo",
		.alg = SPINE_NEO,
	};
	// pr_info("New spine flow, from: %u:%u to %u:%u", dp_info.src_ip,
	// 	dp_info.src_port, dp_info.dst_ip, dp_info.dst_port);
	ca->conn =
		spine_connection_start(kernel_datapath, (void *)sk, &dp_info);
	if (ca->conn == NULL) {
		pr_info("start connection failed\n");
	} else {
		pr_info("starting spine connection %d", ca->conn->index);
	}

	// if no ecn support
	if (!(tp->ecn_flags & TCP_ECN_OK)) {
		INET_ECN_dontxmit(sk);
	}

	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
}

static void neo_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
}

static u32 neo_ssthresh(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	// we want RL to take more efficient control
	struct neo_data *ca = inet_csk_ca(sk);
	u64 rate;

	if (!ca->slow_start_passed){
		ca->slow_start_passed = 1;
		tp->snd_cwnd = tp->snd_cwnd * 717 / 1000;
		neo_update_pacing_rate(sk);
		// ca->last_used_cwnd = cwnd;
		rate = sk->sk_pacing_rate;
		ca->intervals[0].rate = rate;
		ca->ready_rate = rate;
		ca->rate = rate;
	}
	ca->prior_cwnd = tp->snd_cwnd;
	return max(tp->snd_cwnd, 10U);
}

static void neo_set_state(struct sock *sk, u8 new_state)
{	
	struct neo_data *neo = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);
	s32 double_counted;

	if (new_state == TCP_CA_Loss) {
		neo->prev_ca_state = TCP_CA_Loss;
		double_counted = tsk->delivered + tsk->lost+
				 tcp_packets_in_flight(tsk);
		double_counted -= tsk->data_segs_out;
		double_counted -= neo->double_counted;
		neo->double_counted+= double_counted;
		// printk(KERN_INFO "%d loss ended: double_counted %d\n",
		//        neo->id, double_counted);
	}
}

static void neo_pkt_acked(struct sock *sk, const struct ack_sample *sample)
{
}

static u32 neo_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static void slow_set_cwnd(struct sock *sk, u32 acked)
{
	// do_div(change, NEO_SCALE);
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);
	u32 cwnd = tp->snd_cwnd;
	int delta = ca->cnt;
	// printk(KERN_INFO "Delta before division: %d.\n", delta);

	delta = delta / NEO_SCALE;

	if (delta != 0) {
		ca->cnt -= delta * NEO_SCALE;
		// printk(KERN_INFO "[NEO] Old CWND %d, New CWND %d.\n", cwnd, cwnd + delta);
		cwnd += delta;
	}
	cwnd = max(4ULL, cwnd);
	cwnd = min((u32)cwnd, tp->snd_cwnd_clamp); /* apply cap */
	tp->snd_cwnd = cwnd;

}

u32 neo_slow_start(struct sock *sk, u32 acked)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);
	u64 rate;
	ca->cnt += acked * 500;
	slow_set_cwnd(sk, acked);
	neo_update_pacing_rate(sk);

	rate = sk->sk_pacing_rate;
	ca->intervals[0].rate = rate;
	ca->ready_rate = rate;
	ca->rate = rate;
}

static void neo_cong_control(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);
	struct spine_connection *conn = ca->conn;
	u8 prev_state = ca->prev_ca_state, state = inet_csk(sk)->icsk_ca_state;
	u32 acked = rs->acked_sacked; //rs->delivered;
	int ok = 0;
	// printk(KERN_INFO "[NEO] Get into control1.\n");
	// we only do slow start when flow starts
	// if (tcp_in_slow_start(tp) && !ca->slow_start_passed) {
	// 	// printk(KERN_INFO "[NEO] acked: %d, delivered %d.\n, ",  rs->acked_sacked, rs->delivered);
	// 	neo_slow_start(sk, acked);
	// 	goto end;
	// }

	if (prev_state >= TCP_CA_Recovery && state < TCP_CA_Recovery) {
		/* Exiting loss recovery; restore cwnd saved before recovery. */
		tp->snd_cwnd = max(tp->snd_cwnd, ca->prior_cwnd);
	}

	if (rs->delivered < 0 || rs->interval_us < 0) {
		goto end;
	}

	neo_process(sk);
	// printk(KERN_INFO "[NEO] Get into control1.\n");
	// call spine to update parameters if needed
	if (conn != NULL) {
		// if there are staged parameters update, then
		// corressponding params inside ca would be updated
		ok = spine_invoke(conn);
		if (ok < 0) {
			pr_info("fail to call spine_invoke: %d\n", ok);
		}
	}
end:
	ca->lost_base = tp->lost;
	ca->delivered_base = tp->delivered;
}

static struct tcp_congestion_ops neo __read_mostly = {
	.init = neo_init,
	.release = neo_release,
	.ssthresh = neo_ssthresh,
	// .cong_avoid = neo_cong_avoid,
	.cong_control = neo_cong_control,
	.set_state = neo_set_state,
	.undo_cwnd = neo_undo_cwnd,
	// .cwnd_event = neo_cwnd_event,
	.pkts_acked = neo_pkt_acked,
	.owner = THIS_MODULE,
	.name = "neo",
};

static int __init neo_register(void)
{
	int ret;
	BUILD_BUG_ON(sizeof(struct neo_data) > ICSK_CA_PRIV_SIZE);
	ktime_get_real_ts64(&tzero);

	/* Init spine-related structs inspired by CCP
	 * kernel_datapath
	 * spine connections
	 */
	kernel_datapath = (struct spine_datapath *)kmalloc(
		sizeof(struct spine_datapath), GFP_KERNEL);
	if (!kernel_datapath) {
		pr_info("could not allocate spine_datapath\n");
		return -1;
	}
	kernel_datapath->now = &spine_now;
	kernel_datapath->since_usecs = &spine_since;
	kernel_datapath->after_usecs = &spine_after;
	kernel_datapath->log = &spine_log;
	kernel_datapath->fto_us = 1000;
	kernel_datapath->max_connections = MAX_ACTIVE_FLOWS;
	kernel_datapath->spine_active_connections =
		(struct spine_connection *)kzalloc(
			sizeof(struct spine_connection) * MAX_ACTIVE_FLOWS,
			GFP_KERNEL);
	if (!kernel_datapath->spine_active_connections) {
		pr_info("could not allocate spine_connections\n");
		return -2;
	}
	kernel_datapath->log = &spine_log;
	kernel_datapath->set_params = &neo_set_params;
	kernel_datapath->fetch_measurements = &neo_fetch_measurements;
	kernel_datapath->send_msg = &nl_sendmsg;

	/* Here we need to add a IPC for receiving messages from user space 
	 * RL controller.
	 */
	ret = spine_nl_sk(spine_read_msg);
	if (ret < 0) {
		pr_info("cannot init spine ipc\n");
		return -3;
	}
	pr_info("spine ipc init\n");
	// register current sock in spine datapath
	ret = spine_init(kernel_datapath, 0);
	if (ret < 0) {
		pr_info("fail to init spine datapath\n");
		free_spine_nl_sk();
		return -4;
	}
	pr_info("spine %s init\n", neo.name);

	return tcp_register_congestion_control(&neo);
}

static void __exit neo_unregister(void)
{
	free_spine_nl_sk();
	kfree(kernel_datapath->spine_active_connections);
	kfree(kernel_datapath);
	pr_info("spine exit\n");
	tcp_unregister_congestion_control(&neo);
}

module_init(neo_register);
module_exit(neo_unregister);

MODULE_AUTHOR("Han Tian");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("TCP Neo");
MODULE_VERSION("1.0");
