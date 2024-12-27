/* Modified from BytePS */
#ifndef LOGGING_HH
#define LOGGING_HH

#include <sstream>
#include <string>
#include <vector>

enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };

#define LOG_LEVELS "TDIWEF"

// Always-on checking
#define CHECK(x) \
  if (!(x))      \
  common::LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << ' '

#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x)                                     \
  ((x) == NULL ? common::LogMessageFatal(__FILE__, __LINE__) \
                     << "Check  notnull: " #x << ' ',        \
   (x) : (x))  // NOLINT(*)

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

/*
 * \brief Protected NCCL call.
 */
#define NCCLCHECK(cmd)                                                  \
  {                                                                     \
    ncclResult_t r = (cmd);                                             \
    CHECK(r == ncclSuccess) << "NCCL error: " << ncclGetErrorString(r); \
  }

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, LogLevel severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage(bool log_time);

 private:
  const char* fname_;
  int line_;
  LogLevel severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _LOG_TRACE LogMessage(__FILE__, __LINE__, LogLevel::TRACE)
#define _LOG_DEBUG LogMessage(__FILE__, __LINE__, LogLevel::DEBUG)
#define _LOG_INFO LogMessage(__FILE__, __LINE__, LogLevel::INFO)
#define _LOG_WARNING LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _LOG_ERROR LogMessage(__FILE__, __LINE__, LogLevel::ERROR)
#define _LOG_FATAL LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _LOG_##severity

#define _LOG_RANK(severity, rank) _LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();

#endif  // LOGGING_HH
