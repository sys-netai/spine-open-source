#include "spine_priv.h"

#ifdef __KERNEL__
#include <linux/slab.h> // kmalloc
#include <linux/string.h> // memcpy,memset
#else
#include <stdlib.h>
#include <string.h>
#endif

int init_spine_priv_state(struct spine_datapath *datapath,
			  struct spine_connection *conn)
{
	struct spine_priv_state *state;
	conn->state = __CALLOC__(1, sizeof(struct spine_priv_state));
	state = (struct spine_priv_state*) conn->state;
	if (!state) {
		spine_warn(
			"fail to allocate memory for connection private state\n");
		return -1;
	}
	state->sent_create = false;
	conn->datapath = datapath;
	return 1;
}

__INLINE__ void free_spine_priv_state(struct spine_connection *conn)
{
	struct spine_priv_state *state = get_spine_priv_state(conn);
	__FREE__(state);
}
__INLINE__ struct spine_priv_state *
get_spine_priv_state(struct spine_connection *conn)
{
	return (struct spine_priv_state *)conn->state;
}