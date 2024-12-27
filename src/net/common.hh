#ifndef COMMON_HH
#define COMMON_HH

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <sstream>
#include <string>

/* Enable deepCC in kernel */
#define TCP_CWND_CLAMP 42
#define TCP_CWND 43
#define TCP_DEEPCC_ENABLE 44
#define TCP_CWND_CAP 45
#define TCP_DEEPCC_INFO 46 /* Get Congestion Control (optional) orca info */
#define TCP_CWND_MIN 47

#endif /* common */