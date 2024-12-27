#include "tcp_spine.h"
#include "lib/spine.h"
#include "spine_nl.h"

#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/time64.h>
#include <linux/timekeeping.h>
#include <net/tcp.h>

// Global internal state -- allocated during init and freed in free.
struct spine_datapath *kernel_datapath;

void spine_log(struct spine_datapath *dp, enum spine_log_level level,
	       const char *msg, int msg_size)
{
	switch (level) {
	case ERROR:
	case WARN:
	case INFO:
	case DEBUG:
	case TRACE:
		pr_info("%s\n", msg);
		break;
	default:
		break;
	}
}

inline void spine_set_pacing_rate(struct sock *sk, uint32_t rate)
{
	sk->sk_pacing_rate = rate;
}

struct timespec64 tzero;
u64 spine_now(void)
{
	struct timespec64 now, diff;
	ktime_get_real_ts64(&now);
	diff = timespec64_sub(now, tzero);
	return timespec64_to_ns(&diff);
}

u64 spine_since(u64 then)
{
	struct timespec64 now, then_ts, diff;
	ktime_get_real_ts64(&now);
	then_ts = tzero;
	timespec64_add_ns(&then_ts, then);
	diff = timespec64_sub(now, then_ts);
	return timespec64_to_ns(&diff) / NSEC_PER_USEC;
}

u64 spine_after(u64 us)
{
	struct timespec64 now;
	ktime_get_real_ts64(&now);
	now = timespec64_sub(now, tzero);
	timespec64_add_ns(&now, us * NSEC_PER_USEC);
	return timespec64_to_ns(&now);
}
