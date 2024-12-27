#include <linux/gfp.h>
#include <linux/module.h>
#include <linux/netlink.h>
#include <linux/skbuff.h>
#include <net/sock.h>

#include "../src/serialize.h"

#define MYMGRP 22

struct sock *nl_sk = NULL;
void nl_send_msg(unsigned long data, u32 sockid, int pid);
//static struct timer_list timer;

static void log_msg(char *msg, int msg_size)
{
	size_t i;
	char pt[msg_size * 6];
	for (i = 0; i < msg_size; i++) {
		printk(KERN_INFO "msg byte: %lu, %02hhX\n", i, msg[i]);
		snprintf(pt + i * 6, 6, "0x%02hhX, ", msg[i]);
	}

	printk(KERN_INFO "msg: [%s]\n", pt);
}

static void nl_recv_msg(struct sk_buff *skb)
{
	int pid; // pid of sending process
	int sockid;
	int ret;
	//struct sk_buff *skb_out;
	struct nlmsghdr *nlh = nlmsg_hdr(skb);
	struct SpineMsgHeader hdr;
	char *data_ptr;
	//char msg[30];
	int msg_size;
	struct ParamMsg param;
	struct StateMsg state;
	unsigned long reply_num;

	printk(KERN_INFO "Entering %s\n", __FUNCTION__);
	printk(KERN_INFO "Netlink raw rcvd:\n");
	data_ptr = (char *)nlmsg_data(nlh);
	pid = nlh->nlmsg_pid;

	// first read header
	memcpy(&hdr, data_ptr, sizeof(struct SpineMsgHeader));
	sockid = hdr.SocketId;

	printk(KERN_INFO
	       "Netlink received msg type: %d from PID: %d, socket fd: %d\n",
	       hdr.Type, pid, sockid);
	// we begin to parse payload, move pointer
	data_ptr += sizeof(struct SpineMsgHeader);
	if (hdr.Type == STATE) {
		// firstly memset
		memset(&state, 0, sizeof(struct StateMsg));
		memcpy(&state, data_ptr, sizeof(struct StateMsg));
		printk(KERN_INFO
		       "Netlink recv state - number: %d, message: %s\n",
		       state.number, state.message);
		reply_num = state.number + 100;
	} else if (hdr.Type == PARAM) {
		memset(&param, 0, sizeof(struct ParamMsg));
		memcpy(&param, data_ptr, sizeof(struct ParamMsg));
		printk(KERN_INFO
		       "Netlink recv param - cubic_alpha: %d, cubic_beta: %d, message: %s\n",
		       param.cubic_alpha, param.cubic_beta, param.message);
		reply_num = param.cubic_alpha + 100;
	} else {
		printk(KERN_INFO, "unknown message type\n");
		return -1;
	}
	// kernel reply with one number
	nl_send_msg(reply_num, sockid, pid);
}

void nl_send_msg(unsigned long data, u32 sockid, int pid)
{
	struct sk_buff *skb_out;
	struct nlmsghdr *nlh = NULL;

	int res;
	char message[50];
	int reply_size;
	reply_size = sizeof(struct SpineMsgHeader) + sizeof(struct ParamMsg);
	char buf[reply_size];

	struct SpineMsgHeader header = {
		.Len = reply_size,
		.Type = STATE,
		.SocketId = sockid,
	};
	struct StateMsg ms = {
		.number = data,
	};
	// be careful when dealing with char array.
	memset(message, 0, 50);
	snprintf(message, 50, "hello from kernel, number: %lu!", data);
	memcpy(ms.message, message, 50);

	// copy header and message to buffer
	memset(buf, 0, reply_size);
	memcpy(buf, &header, sizeof(struct SpineMsgHeader));

	memcpy(buf + sizeof(struct SpineMsgHeader), &ms,
	       sizeof(struct StateMsg));

	skb_out = nlmsg_new(
		NLMSG_ALIGN(
			reply_size), // @payload: size of the message payload
		GFP_KERNEL // @flags: the type of memory to allocate.
	);
	if (!skb_out) {
		printk(KERN_INFO "Failed to allocate new skb\n");
		return;
	}

	nlh = nlmsg_put(skb_out, // @skb: socket buffer to store message in
			0, // @portid: netlink PORTID of requesting application
			0, // @seq: sequence number of message
			NLMSG_DONE, // @type: message type
			reply_size, // @payload: length of message payload
			0 // @flags: message flags
	);

	if (nlh == NULL) {
		printk(KERN_INFO "Failed to allocate new nl_header\n");
		nlmsg_free(skb_out);
		return;
	}

	//NETLINK_CB(skb_out).dst_group = 0;
	memcpy(nlmsg_data(nlh), buf, reply_size);
	printk(KERN_INFO "Sending proactive kernel message\n");
	res = nlmsg_multicast(
		nl_sk, // @sk: netlink socket to spread messages to
		skb_out, // @skb: netlink message as socket buffer
		0, // @portid: own netlink portid to avoid sending to yourself
		MYMGRP, // @group: multicast group id
		GFP_KERNEL // @flags: allocation flags
	);
	// res = nlmsg_unicast(nl_sk, // @sk: netlink socket to spread messages to
	// 		    skb_out, // @skb: netlink message as socket buffer
	// 		    pid);
	if (res < 0) {
		printk(KERN_INFO "Error while sending to user (pid: %d): %d\n",
		       pid, res);
		/* Wait 1 second. */
		//mod_timer(&timer, jiffies + msecs_to_jiffies(1000));
	} else {
		printk(KERN_INFO "Send ok\n");
	}
}

static int __init nl_init(void)
{
	struct netlink_kernel_cfg cfg = {
		.input = nl_recv_msg,
	};

	printk(KERN_INFO "init NL\n");

	nl_sk = netlink_kernel_create(&init_net, NETLINK_USERSOCK, &cfg);
	if (!nl_sk) {
		printk(KERN_ALERT "Error creating socket.\n");
		return -10;
	}

	//init_timer(&timer);
	//timer.function = nl_send_msg;
	//timer.expires = jiffies + 1000;
	//timer.data = 0;
	//add_timer(&timer);
	return 0;
}

static void __exit nl_exit(void)
{
	printk(KERN_INFO "exit NL\n");
	//del_timer_sync(&timer);
	netlink_kernel_release(nl_sk);
}

module_init(nl_init);
module_exit(nl_exit);

MODULE_LICENSE("GPL");
