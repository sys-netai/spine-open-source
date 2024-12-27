#ifndef SPINE_ERR_H
#define SPINE_ERR_H

#define SPINE_OK 1

// regular err
#define SPINE_ERROR -1

// Function parameter checking
#define SPINE_MISSING_ARG -11
#define SPINE_NULL_ARG -12

// Buffer size checking
#define SPINE_BUFSIZE_NEGATIVE -21
#define SPINE_BUFSIZE_TOO_SMALL -22
#define SPINE_MSG_TOO_LONG -23

// Se/deserializing messages
#define SPINE_WRITE_INVALID_HEADER_TYPE -31
#define SPINE_READ_INVALID_HEADER_TYPE -32
#define SPINE_READ_INVALID_OP -33
#define SPINE_READ_REG_NOT_ALLOWED -34
#define SPINE_READ_INVALID_RETURN_REG -35
#define SPINE_READ_INVALID_LEFT_REG -36
#define SPINE_READ_INVALID_RIGHT_REG -37

// Install message parse errors
#define SPINE_INSTALL_TYPE_MISMATCH -41
#define SPINE_INSTALL_TOO_MANY_EXPR -42
#define SPINE_INSTALL_TOO_MANY_INSTR -43

// Update message parse errors
#define SPINE_UPDATE_TYPE_MISMATCH -51
#define SPINE_UPDATE_TOO_MANY -52
#define SPINE_UPDATE_INVALID_REG_TYPE -53

// Change message parse errors
#define SPINE_CHANGE_TYPE_MISMATCH -61
#define SPINE_CHANGE_TOO_MANY -62

// Connection object
#define SPINE_UNKNOWN_CONNECTION -71
#define SPINE_CREATE_PENDING -72
#define SPINE_CONNECTION_NOT_INITIALIZED -73

// Datapath programs
#define SPINE_PROG_TABLE_FULL -81
#define SPINE_PROG_NOT_FOUND -82

// VM instruction execution errors
#define SPINE_ADD_INT_OVERFLOW -91
#define SPINE_DIV_BY_ZERO -92
#define SPINE_MUL_INT_OVERFLOW -93
#define SPINE_SUB_INT_UNDERFLOW -94
#define SPINE_PRIV_IS_NULL -95
#define SPINE_PROG_IS_NULL -96

// Fallback timer
#define SPINE_FALLBACK_TIMED_OUT -101


#endif