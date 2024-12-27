#ifndef SPINE_NL_H
#define SPINE_NL_H

#include "lib/spine.h"

/* -- netlink related --- */
typedef int (*spine_nl_recv_handler)(struct spine_datapath *datapath, char *msg,
				     int msg_size);

/* Create a netlink kernel socket
 * A global (struct sock*), ccp_nl_sk, will get set so we can use the socket
 * There is *only one* netlink socket active *per datapath*
 */
int spine_nl_sk(spine_nl_recv_handler msg);

/* Wrap netlink_kernel_release of (struct sock *ccp_nl_sk).
 */
void free_spine_nl_sk(void);

/* Send serialized message to userspace CCP
 */
int nl_sendmsg(struct spine_datapath *dp, char *msg, int msg_size);

#endif