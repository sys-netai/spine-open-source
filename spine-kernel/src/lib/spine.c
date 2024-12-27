#ifdef __KERNEL__
#include <linux/slab.h> // kmalloc
#include <linux/string.h> // memcpy
#include <linux/types.h>
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "spine.h"
#include "spine_err.h"
#include "spine_priv.h"

#define CREATE_TIMEOUT_US 100000 // 100 ms

int spine_init(struct spine_datapath *datapath, u32 id)
{
	int ok;
	char ready_msg[READY_MSG_SIZE];
	spine_trace("spine init");
	if (datapath == NULL || datapath->set_params == NULL ||
	    datapath->now == NULL || datapath->since_usecs == NULL ||
	    datapath->after_usecs == NULL ||
	    datapath->spine_active_connections == NULL ||
	    datapath->max_connections == 0 || datapath->fto_us == 0 || datapath->send_msg == NULL) {
		return SPINE_MISSING_ARG;
	}
	// send ready message
	ok = write_ready_msg(ready_msg, READY_MSG_SIZE, id);
	if (ok < 0) {
		spine_error("could not serialize ready message") return ok;
	}

	ok = datapath->send_msg(datapath, ready_msg, READY_MSG_SIZE);
	if (ok < 0) {
		spine_warn("could not send ready message: %d", ok)
	}

	spine_trace("wrote ready msg");
	datapath->time_zero = datapath->now();
	datapath->last_msg_sent = 0;
	datapath->_in_fallback = false;
	return SPINE_OK;
}

void spine_free(struct spine_datapath *datapath) {
	(void)(datapath);
}

void spine_conn_create_success(struct spine_priv_state *state)
{
	state->sent_create = true;
}

__INLINE__ void *spine_get_impl(struct spine_connection *conn)
{
	return conn->impl;
}

__INLINE__ void spine_set_impl(struct spine_connection *conn, void *ptr)
{
	conn->impl = ptr;
}

struct spine_connection *
spine_connection_start(struct spine_datapath *datapath, void *impl,
		       struct spine_datapath_info *flow_info)
{
	int ret;
	u16 sid;
	struct spine_connection *conn;

	// scan to find empty place
	// index = 0 means free/unused
	for (sid = 0; sid < datapath->max_connections; sid++) {
		conn = &datapath->spine_active_connections[sid];
		if (CAS(&(conn->index), 0, sid + 1)) {
			sid = sid + 1;
			break;
		}
	}

	if (sid >= datapath->max_connections) {
		return NULL;
	}

	conn->impl = impl;
	memcpy(&conn->flow_info, flow_info, sizeof(struct spine_datapath_info));

	// init connection private state
	ret = init_spine_priv_state(datapath, conn);

	if (ret < 0) {
		spine_error("could not init connection private state");
		return NULL;
	}

	// send to CCP:
	// index of pointer back to this sock for IPC callback
	// TODO implement this function
	ret = send_conn_create(datapath, conn);
	if (ret < 0) {
		if (!datapath->_in_fallback) {
			spine_warn("failed to send create message: %d\n", ret);
		}
		return conn;
	}
	struct spine_priv_state *state = get_spine_priv_state(conn);
    spine_conn_create_success(state);
	return conn;
}

struct spine_connection *
spine_connection_lookup(struct spine_datapath *datapath, u16 sid)
{
	struct spine_connection *conn;
	// bounds check
	if (sid == 0 || sid > datapath->max_connections) {
		spine_warn("index out of bounds: %d", sid);
		return NULL;
	}

	conn = &datapath->spine_active_connections[sid - 1];
	if (conn->index != sid) {
		spine_trace("index mismatch: sid %d, index %d", sid,
			    conn->index);
		return NULL;
	}

	return conn;
}

void spine_connection_free(struct spine_datapath *datapath, u16 sid)
{
	int msg_size, ret;
	struct spine_connection *conn;
	char msg[REPORT_MSG_SIZE];

	spine_trace("Entering %s\n", __FUNCTION__);
	// bounds check
	if (sid == 0 || sid > datapath->max_connections) {
		spine_warn("index out of bounds: %d", sid);
		return;
	}

	conn = &datapath->spine_active_connections[sid - 1];
	if (conn->index != sid) {
		spine_warn("index mismatch: sid %d, index %d", sid,
			   conn->index);
		return;
	}

	free_spine_priv_state(conn);

	msg_size = write_measure_msg(msg, REPORT_MSG_SIZE, sid, 0, 0, 0);
	ret = datapath->send_msg(datapath, msg, msg_size);
	if (ret < 0) {
		if (!datapath->_in_fallback) {
			spine_warn("error sending close message: %d", ret);
		}
	}

	// spine_connection_start will look for an array entry with index 0
	// to indicate that it's available for a new flow's information.
	// So, we set index to 0 here to reuse the memory.
	conn->index = 0;
	return;
}


int spine_invoke(struct spine_connection *conn)
{
	int ret = 0;
	int i;
	struct spine_priv_state *state;
	struct spine_datapath *datapath;
	u8 num_params = 0;
	u64 params[MAX_CONTROL_REG];
	
	spine_trace("Entering %s\n", __FUNCTION__);
	if (conn == NULL) {
		return SPINE_NULL_ARG;
	}
	datapath = conn->datapath;
	state = get_spine_priv_state(conn);

	if (!state) {
		spine_error("no private state for connection");
		return SPINE_NULL_ARG;
	}

	// check init status
	if (!(state->sent_create)) {
        // try contacting the CCP again
        // index of pointer back to this sock for IPC callback
        spine_trace("%s retx create message\n", __FUNCTION__);
        ret = send_conn_create(datapath, conn);
        if (ret < 0) {
            if (!datapath->_in_fallback) {
                spine_warn("failed to retx create message: %d\n", ret);
            }
        } else {
            spine_conn_create_success(state);
        }
        return SPINE_OK;
    }

	// we assume consequent parameters
	for (i = 0; i < MAX_CONTROL_REG; i++) {
		if (state->pending_update.control_is_pending[i]) {
			params[i] = state->pending_update.control_registers[i];
			num_params += 1;
		} else {
			// there are no remaining staged parameters, we stop here
			break;
		}
	}
	// enforce parameters to datapath
	if (num_params > 0 && datapath->set_params) {
		datapath->set_params(conn, params, num_params);
	}
	// clear staged status
	memset(&state->pending_update, 0, sizeof(struct staged_update));
	return ret;
}

int send_conn_create(struct spine_datapath *datapath,
		     struct spine_connection *conn)
{
	int ret;
	char msg[CREATE_MSG_SIZE];
	int msg_size;
	struct CreateMsg cr = {
		.init_cwnd = conn->flow_info.init_cwnd,
		.mss = conn->flow_info.mss,
		.src_ip = conn->flow_info.src_ip,
		.src_port = conn->flow_info.src_port,
		.dst_ip = conn->flow_info.dst_ip,
		.dst_port = conn->flow_info.dst_port,
	};
	memcpy(&cr.congAlg, &conn->flow_info.congAlg, MAX_CONG_ALG_SIZE);

	if (conn->last_create_msg_sent != 0 &&
	    datapath->since_usecs(conn->last_create_msg_sent) <
		    CREATE_TIMEOUT_US) {
		spine_trace("%s: " FMT_U64 " < " FMT_U32 "\n", __FUNCTION__,
			    datapath->since_usecs(conn->last_create_msg_sent),
			    CREATE_TIMEOUT_US);
		return SPINE_CREATE_PENDING;
	}

	if (conn->index < 1) {
		return SPINE_CONNECTION_NOT_INITIALIZED;
	}

	conn->last_create_msg_sent = datapath->now();
	msg_size = write_create_msg(msg, CREATE_MSG_SIZE, conn->index, cr);
	if (msg_size < 0) {
		return msg_size;
	}

	ret = datapath->send_msg(datapath, msg, msg_size);
	if (ret) {
		spine_debug("error sending create, updating fto_timer");
		_update_fto_timer(datapath);
	}
	return ret;
}


int stage_update(struct spine_datapath *datapath __attribute__((unused)),
		 struct staged_update *pending_update,
		 struct UpdateField *update_field)
{
	// update the value for these registers
	// for cwnd, rate; update field in datapath
	switch (update_field->reg_type) {
	case NONVOLATILE_CONTROL_REG:
	case VOLATILE_CONTROL_REG:
		// set new value
		spine_trace(("%s: control " FMT_U32 " <- " FMT_U64 "\n"),
			    __FUNCTION__, update_field->reg_index,
			    update_field->new_value);
		pending_update->control_registers[update_field->reg_index] =
			update_field->new_value;
		pending_update->control_is_pending[update_field->reg_index] =
			true;
		return SPINE_OK;
	default:
		return SPINE_UPDATE_INVALID_REG_TYPE; // allowed only for CONTROL and CWND and RATE reg within CONTROL_REG
	}
}

int stage_multiple_updates(struct spine_datapath *datapath,
			   struct staged_update *pending_update,
			   size_t num_updates, struct UpdateField *msg_ptr)
{
	int ret;
	for (size_t i = 0; i < num_updates; i++) {
		ret = stage_update(datapath, pending_update, msg_ptr);
		if (ret < 0) {
			return ret;
		}

		msg_ptr++;
	}

	return SPINE_OK;
}

/* Read parameters from user-space RL algorithm
 * first save these parameters to staged registers
 */
int spine_read_msg(struct spine_datapath *datapath, char *buf, int bufsize)
{
	// TODO: Implement semantics to read control message from user space
	int ret;
	int msg_program_index;
	u32 num_updates;
	char *msg_ptr;
	struct SpineMsgHeader hdr;
	struct spine_connection *conn;
	struct spine_priv_state *state;

	ret = read_header(&hdr, buf);
	if (ret < 0) {
		spine_warn("read header failed: %d", ret);
		return ret;
	}

	if (bufsize < 0) {
		spine_warn("negative bufsize: %d", bufsize);
		return SPINE_BUFSIZE_NEGATIVE;
	}
	if (hdr.Len > ((u32)bufsize)) {
		spine_warn("message size wrong: %u > %d\n", hdr.Len, bufsize);
		return SPINE_BUFSIZE_TOO_SMALL;
	}

	if (hdr.Len > BIGGEST_MSG_SIZE) {
		spine_warn("message too long: %u > %d\n", hdr.Len,
			   BIGGEST_MSG_SIZE);
		return SPINE_MSG_TOO_LONG;
	}
	msg_ptr = buf + ret;

	_turn_off_fto_timer(datapath);

	// rest of the messages must be for a specific flow
	conn = spine_connection_lookup(datapath, hdr.SocketId);
	if (conn == NULL) {
		spine_trace("unknown connection: %u\n", hdr.SocketId);
		return SPINE_UNKNOWN_CONNECTION;
	}
	state = get_spine_priv_state(conn);

	// here we only need this
	if (hdr.Type == UPDATE_FIELDS) {
		spine_debug("[sid=%d] Received update_fields message\n",
			    conn->index);
		ret = check_update_fields_msg(datapath, &hdr, &num_updates,
					      msg_ptr);
		msg_ptr += ret;
		if (ret < 0) {
			spine_warn("Update fields message failed: %d\n", ret);
			return ret;
		}

		ret = stage_multiple_updates(datapath, &state->pending_update,
					     num_updates,
					     (struct UpdateField *)msg_ptr);
		if (ret < 0) {
			spine_warn(
				"update_fields: failed to stage updates: %d\n",
				ret);
			return ret;
		}

		spine_debug("Staged %u updates\n", num_updates);
	} 
	return SPINE_OK;
}

void _update_fto_timer(struct spine_datapath *datapath)
{
	if (!datapath->last_msg_sent) {
		datapath->last_msg_sent = datapath->now();
	}
}

/*
 * Returns true if CCP has timed out, false otherwise
 */
bool _check_fto(struct spine_datapath *datapath)
{
	// TODO not sure how well this will scale with many connections,
	//      may be better to make it per conn
	u64 since_last = datapath->since_usecs(datapath->last_msg_sent);
	bool should_be_in_fallback =
		datapath->last_msg_sent && (since_last > datapath->fto_us);

	if (should_be_in_fallback && !datapath->_in_fallback) {
		datapath->_in_fallback = true;
		spine_error("spine fallback (%lu since last msg)\n",
			    since_last);
	} else if (!should_be_in_fallback && datapath->_in_fallback) {
		datapath->_in_fallback = false;
		spine_error("spine should not be in fallback");
	}
	return should_be_in_fallback;
}

void _turn_off_fto_timer(struct spine_datapath *datapath)
{
	if (datapath->_in_fallback) {
		spine_error("spine restored!\n");
	}
	datapath->_in_fallback = false;
	datapath->last_msg_sent = 0;
}