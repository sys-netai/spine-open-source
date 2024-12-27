/* 
 * CCP Datapath Message Serialization 
 * 
 * Serializes and deserializes messages for communication with userspace CCP.
 */
#ifndef CCP_SERIALIZE_H
#define CCP_SERIALIZE_H

#include "spine.h"

#define LIBCCP_BUFSIZE_TOO_SMALL -1
#define LIBCCP_BUFSIZE_NEGATIVE -1
#define LIBCCP_INSTALL_TYPE_MISMATCH -1
#define LIBCCP_INSTALL_TOO_MANY_EXPR -1
#define LIBCCP_INSTALL_TOO_MANY_INSTR -1
#define LIBCCP_UPDATE_TOO_MANY -1
#define LIBCCP_CHANGE_TOO_MANY -1
#define LIBCCP_UPDATE_TYPE_MISMATCH -1

#ifdef __cplusplus
extern "C" {
#endif

struct __attribute__((packed, aligned(4))) SpineMsgHeader {
	u16 Type;
	u16 Len;
	u32 SocketId;
};

/* return: sizeof(struct CcpMsgHeader) on success, -1 otherwise.
 */
int read_header(struct SpineMsgHeader *hdr, char *buf);

/* return: sizeof(struct CcpMsgHeader) on success, -1 otherwise.
 */
int serialize_header(char *buf, int bufsize, struct SpineMsgHeader *hdr);

/* There are 3 message types (Type field in header)
 * CREATE and MEASURE are written from datapath to CCP
 * PATTERN and INSTALL_FOLD are received in datapath from CCP
 * 
 * Messages start with the header, then 
 * 1. fixed number of u32
 * 2. fixed number of u64
 * 3. bytes blob, flexible length
 */
#define CREATE 0
#define MEASURE 1
#define INSTALL_EXPR 2
#define UPDATE_FIELDS 3
#define CHANGE_PROG 4
#define READY 5

// spine message types
#define STATE 6
#define PARAM 7
#define NEURAL_NETWORK 8

// Some messages contain strings.
#define BIGGEST_MSG_SIZE 32678

// create messages are fixed length: header + 4 * 6 + 32
#define CREATE_MSG_SIZE 96
// size of report msg is approx MAX_REPORT_REG * 8 + 4 + 4
#define REPORT_MSG_SIZE 900
// ready message is just a u32.
#define READY_MSG_SIZE 12

// Some messages contain serialized fold instructions.
#define MAX_EXPRESSIONS 256 // arbitrary TODO: make configurable
#define MAX_INSTRUCTIONS 256 // arbitrary, TODO: make configurable
#define MAX_IMPLICIT_REG 6 // fixed number of implicit registers
#define MAX_REPORT_REG 110 // measure msg 110 * 8 + 4 + 4
#define MAX_CONTROL_REG 110 // arbitrary
#define MAX_TMP_REG 8
#define MAX_LOCAL_REG 8
#define MAX_MUTABLE_REG 222 // # report + # control + cwnd, rate registers

struct __attribute__((packed, aligned(4))) StateMsg {
	u32 number;
	char message[64];
};

struct __attribute__((packed, aligned(4))) ParamMsg {
	u32 cubic_alpha;
	u32 cubic_beta;
	char message[64];
};

struct __attribute__((packed, aligned(4))) ReadyMsg {
	u32 id;
};

/* READY
 * id: The unique id of this datapath.
 */
int write_ready_msg(char *buf, int bufsize, u32 id);

/* CREATE
 * congAlg: the datapath's requested congestion control algorithm (could be overridden)
 */
struct __attribute__((packed, aligned(4))) CreateMsg {
	u32 init_cwnd;
	u32 mss;
	u32 src_ip;
	u32 src_port;
	u32 dst_ip;
	u32 dst_port;
	char congAlg[MAX_CONG_ALG_SIZE];
};

/* Write cr: CreateMsg into buf with socketid sid.
 * buf should be preallocated, and bufsize should be its size.
 */
int write_create_msg(char *buf, int bufsize, u32 sid, struct CreateMsg cr);

/* MEASURE
 * program_uid: unique id for the datapath program that generated this report,
 *              so that the ccp can use the corresponding scope
 * num_fields: number of returned fields,
 * bytes: the return registers of the installed fold function ([]uint64).
 *        there will be at most MAX_PERM_REG returned registers
 */
struct __attribute__((packed, aligned(4))) MeasureMsg {
	u32 program_uid;
	u32 num_fields;
	u64 fields[MAX_REPORT_REG];
};

/* Write ms: MeasureMsg into buf with socketid sid.
 * buf should be preallocated, and bufsize should be its size.
 */
int write_measure_msg(char *buf, int bufsize, u32 sid, u32 program_uid,
		      u64 *msg_fields, u8 num_fields);


struct __attribute__((packed, aligned(1))) UpdateField {
	u8 reg_type;
	u32 reg_index;
	u64 new_value;
};

/* Fills in number of updates.
 * Check whether number of updates is too large.
 * Returns size of update field header: 1 u32
 * UpdateFieldsMsg:
 * {
 *  1 u32: num_updates
 *  UpdateField[num_updates]
 * }
 */
int check_update_fields_msg(struct spine_datapath *datapath,
			    struct SpineMsgHeader *hdr, u32 *num_updates,
			    char *buf);

struct __attribute__((packed, aligned(1))) ChangeProgMsg {
	u32 program_uid;
	u32 num_updates;
};

int read_change_prog_msg(struct spine_datapath *datapath,
			 struct SpineMsgHeader *hdr,
			 struct ChangeProgMsg *change_prog, char *buf);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
