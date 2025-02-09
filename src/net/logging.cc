#include "logging.hh"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>

LogMessage::LogMessage(const char* fname, int line, LogLevel severity)
    : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::GenerateLogMessage(bool log_time) {
  // we want all logs to go to stderr
  // original level to use stderr is INFO
  bool use_cout =
      static_cast<int>(severity_) < static_cast<int>(LogLevel::TRACE);
  // std::ostream& os = use_cout ? std::cout : std::cerr;
  std::stringstream stream;
  auto os = use_cout ? stdout : stderr;
  if (log_time) {
    auto now = std::chrono::system_clock::now();
    auto as_time_t = std::chrono::system_clock::to_time_t(now);

    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto micros_remainder =
        std::chrono::duration_cast<std::chrono::microseconds>(duration -
                                                              seconds);

    const size_t time_buffer_size = 30;
    char time_buffer[time_buffer_size];
    strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
             localtime(&as_time_t));
    stream << "[" << time_buffer << "." << std::setw(6) << micros_remainder.count()
       << ": " << LOG_LEVELS[static_cast<int>(severity_)] << " " << fname_
       << ":" << line_ << "] " << str() << std::endl;
  } else {
    stream << "[" << LOG_LEVELS[static_cast<int>(severity_)] << " " << fname_ << ":"
       << line_ << "] " << str() << std::endl;
  }
  // dump stream.str to stdout/stderr
  fprintf(os, stream.str().c_str());
}

LogMessage::~LogMessage() {
  static LogLevel min_log_level = MinLogLevelFromEnv();
  static bool log_time = LogTimeFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage(log_time);
  }
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, LogLevel::FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  static bool log_time = LogTimeFromEnv();
  GenerateLogMessage(log_time);
  abort();
}

LogLevel ParseLogLevelStr(const char* env_var_val) {
  std::string min_log_level(env_var_val);
  std::transform(min_log_level.begin(), min_log_level.end(),
                 min_log_level.begin(), ::tolower);
  if (min_log_level == "trace") {
    return LogLevel::TRACE;
  } else if (min_log_level == "debug") {
    return LogLevel::DEBUG;
  } else if (min_log_level == "info") {
    return LogLevel::INFO;
  } else if (min_log_level == "warning") {
    return LogLevel::WARNING;
  } else if (min_log_level == "error") {
    return LogLevel::ERROR;
  } else if (min_log_level == "fatal") {
    return LogLevel::FATAL;
  } else {
    return LogLevel::WARNING;
  }
}

LogLevel MinLogLevelFromEnv() {
  const char* env_var_val = getenv("LOG_LEVEL");
  if (env_var_val == nullptr) {
    // default to WARNING
    return LogLevel::WARNING;
  }
  return ParseLogLevelStr(env_var_val);
}

bool LogTimeFromEnv() {
  const char* env_var_val = getenv("LOG_HIDE_TIME");
  if (env_var_val != nullptr && std::strtol(env_var_val, nullptr, 10) > 0) {
    return false;
  } else {
    return true;
  }
}
