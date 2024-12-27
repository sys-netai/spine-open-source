#include "serialize.h"
#include "spine.h"
#include "spine_err.h"
#include "spine_priv.h"

#ifdef __KERNEL__
#include <linux/slab.h> // kmalloc
#include <linux/string.h> // memcpy
#include <linux/types.h>
#else
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

/* (type, len, socket_id) header
 * -----------------------------------
 * | Msg Type | Len (2B) | Uint32    |
 * | (2 B)    | (2 B)    | (32 bits) |
 * -----------------------------------
 * total: 6 Bytes
 */

/* We only read Install, Update, and Change Program messages.
 */
int read_header(struct SpineMsgHeader *hdr, char *buf)
{
	memcpy(hdr, buf, sizeof(struct SpineMsgHeader));

	switch (hdr->Type) {
	case STATE:
		return sizeof(struct SpineMsgHeader);
	case PARAM:
		return sizeof(struct SpineMsgHeader);
	case UPDATE_FIELDS:
		return sizeof(struct SpineMsgHeader);
	case MEASURE:
		return sizeof(struct SpineMsgHeader);
	default:
		return -1;
	}
}

/* We only write Create, Ready, and Measure messages.
 */
int serialize_header(char *buf, int bufsize, struct SpineMsgHeader *hdr)
{
	switch (hdr->Type) {
	case CREATE:
	case STATE:
	case MEASURE:
	case READY:
	case RELEASE:
		break;
	default:
		printk("[spine] Unknown message type, cannot serialize header");
		return -1;
	}

	if (bufsize < ((int)sizeof(struct SpineMsgHeader))) {
		return LIBCCP_BUFSIZE_TOO_SMALL;
	}

	memcpy(buf, hdr, sizeof(struct SpineMsgHeader));
	return sizeof(struct SpineMsgHeader);
}

int write_ready_msg(char *buf, int bufsize, u32 id)
{
	struct SpineMsgHeader hdr;
	int ret;
	u16 msg_len = sizeof(struct SpineMsgHeader) + sizeof(u32);

	hdr = (struct SpineMsgHeader){ .Type = READY,
				       .Len = msg_len,
				       .SocketId = 0 };

	if (bufsize < 0) {
		return LIBCCP_BUFSIZE_NEGATIVE;
	}

	if (((u32)bufsize) < hdr.Len) {
		return LIBCCP_BUFSIZE_TOO_SMALL;
	}

	ret = serialize_header(buf, bufsize, &hdr);
	if (ret < 0) {
		return ret;
	}

	buf += ret;
	memcpy(buf, &id, sizeof(u32));
	return hdr.Len;
}

int write_release_msg(char *buf, int bufsize, u32 id)
{
	struct SpineMsgHeader hdr;
	int ret;
	u16 msg_len = sizeof(struct SpineMsgHeader) + sizeof(u32);

	hdr = (struct SpineMsgHeader){ .Type = RELEASE,
				       .Len = msg_len,
				       .SocketId = 0 };

	if (bufsize < 0) {
		return LIBCCP_BUFSIZE_NEGATIVE;
	}

	if (((u32)bufsize) < hdr.Len) {
		return LIBCCP_BUFSIZE_TOO_SMALL;
	}

	ret = serialize_header(buf, bufsize, &hdr);
	if (ret < 0) {
		return ret;
	}

	buf += ret;
	memcpy(buf, &id, sizeof(u32));
	return hdr.Len;
}

int write_create_msg(char *buf, int bufsize, u32 sid, struct CreateMsg cr)
{
	struct SpineMsgHeader hdr;
	int ret;
	u16 msg_len = sizeof(struct SpineMsgHeader) + sizeof(struct CreateMsg);

	hdr = (struct SpineMsgHeader){
		.Type = CREATE,
		.Len = msg_len,
		.SocketId = sid,
	};

	if (bufsize < 0) {
		return LIBCCP_BUFSIZE_NEGATIVE;
	}

	if (((u32)bufsize) < hdr.Len) {
		return LIBCCP_BUFSIZE_TOO_SMALL;
	}

	ret = serialize_header(buf, bufsize, &hdr);
	if (ret < 0) {
		return ret;
	}

	buf += ret;
	memcpy(buf, &cr, hdr.Len - sizeof(struct SpineMsgHeader));
	return hdr.Len;
}

int write_measure_msg(char *buf, int bufsize, u32 sid, u32 program_uid,
		      u64 *msg_fields, u8 num_fields)
{
	int ret;
	struct MeasureMsg ms = {
		/* actually this program_uid is echoed request id */
		.program_uid = program_uid,
		.num_fields = num_fields,
	};

	// 4 bytes for num_fields (u32) and 4 for program_uid = 8
	u16 msg_len =
		sizeof(struct SpineMsgHeader) + 8 + ms.num_fields * sizeof(u64);
	struct SpineMsgHeader hdr = {
		.Type = MEASURE,
		.Len = msg_len,
		.SocketId = sid,
	};

	// copy message fields into MeasureMsg struct
	if (msg_fields) {
		memcpy(ms.fields, msg_fields, ms.num_fields * sizeof(u64));
	}

	if (bufsize < 0) {
		return LIBCCP_BUFSIZE_NEGATIVE;
	}

	if (((u32)bufsize) < hdr.Len) {
		return LIBCCP_BUFSIZE_TOO_SMALL;
	}

	ret = serialize_header(buf, bufsize, &hdr);
	if (ret < 0) {
		return ret;
	}

	buf += ret;
	memcpy(buf, &ms, hdr.Len - sizeof(struct SpineMsgHeader));
	return hdr.Len;
}

int check_update_fields_msg(struct spine_datapath *datapath,
			    struct SpineMsgHeader *hdr, u32 *num_updates,
			    char *buf)
{
	if (hdr->Type != UPDATE_FIELDS) {
		spine_warn(
			"check_update_fields_msg: hdr.Type != UPDATE_FIELDS");
		return LIBCCP_UPDATE_TYPE_MISMATCH;
	}

	*num_updates = (u32)*buf;
	if (*num_updates > MAX_MUTABLE_REG) {
		spine_warn("Too many updates!: %u\n", *num_updates);
		return LIBCCP_UPDATE_TOO_MANY;
	}
	return sizeof(u32);
}
int check_measure_fields_msg(struct spine_datapath *datapath,
			     struct SpineMsgHeader *hdr, u32 *measure_idx,
			     char *buf)
{
	if (hdr->Type != MEASURE) {
		spine_warn("check_measure_fields_msg: hdr.Type != MEASURE");
		return LIBCCP_UPDATE_TYPE_MISMATCH;
	}

	*measure_idx = (u32)*buf;
	if (*measure_idx < 0) {
		spine_warn("try to fecth invalid measurements") return -1;
	}
	return sizeof(u32);
}
