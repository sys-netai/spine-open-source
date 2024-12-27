/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include "file_descriptor.hh"

#include <fcntl.h>
#include <unistd.h>

#include <cassert>

#include "exception.hh"

using namespace std;

/* construct from fd number */
FileDescriptor::FileDescriptor(const int fd)
    : fd_(fd), eof_(false), read_count_(0), write_count_(0) {
  if (fd_ <= 2) { /* make sure not overwriting stdout/stderr */
    throw unix_error("FileDescriptor: fd <= 2");
  }

  /* set close-on-exec flag so our file descriptors
     aren't passed on to unrelated children (like a shell) */
  SystemCall("fcntl FD_CLOEXEC", fcntl(fd_, F_SETFD, FD_CLOEXEC));
}

/* move constructor */
FileDescriptor::FileDescriptor(FileDescriptor&& other)
    : fd_(other.fd_),
      eof_(other.eof_),
      read_count_(other.read_count_),
      write_count_(other.write_count_) {
  /* mark other file descriptor as inactive */
  other.fd_ = -1;
}

/* move assignment */
FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) {
  fd_ = other.fd_;
  eof_ = other.eof_;
  read_count_ = other.read_count_;
  write_count_ = other.write_count_;

  /* mark other file descriptor as inactive */
  other.fd_ = -1;

  return *this;
}

/* close method throws exception on failure */
void FileDescriptor::close() {
  if (fd_ < 0) { /* has already been moved away or closed */
    return;
  }

  SystemCall("close", ::close(fd_));

  fd_ = -1;
}

/* destructor */
FileDescriptor::~FileDescriptor() {
  if (fd_ < 0) { /* has already been moved away */
    return;
  }

  try {
    close();
  } catch (const exception& e) { /* don't throw from destructor */
    print_exception("FileDescriptor", e);
  }
}

/* attempt to write a portion of a string */
string::const_iterator FileDescriptor::write(
    const string::const_iterator& begin, const string::const_iterator& end) {
  if (begin >= end) {
    throw runtime_error("nothing to write");
  }

  ssize_t bytes_written =
      SystemCall("write", ::write(fd_, &*begin, end - begin));
  if (bytes_written == 0) {
    throw runtime_error("write returned 0");
  }

  register_write();

  return begin + bytes_written;
}

/* read method */
string FileDescriptor::read(const size_t limit) {
  char buffer[BUFFER_SIZE];

  ssize_t bytes_read =
      SystemCall("read", ::read(fd_, buffer, min(BUFFER_SIZE, limit)));
  if (bytes_read == 0) {
    set_eof();
  }

  register_read();

  return string(buffer, bytes_read);
}

/* read exactly some bytes */
string FileDescriptor::read_exactly(const size_t length,
                                    const bool fail_silently) {
  std::string ret;

  while (ret.size() < length) {
    ret.append(read(length - ret.size()));
    if (eof()) {
      if (fail_silently) {
        return ret;
      } else {
        throw std::runtime_error(
            "read_exactly: reached EOF before reaching target");
      }
    }
  }

  assert(ret.size() == length);
  return ret;
}

/* write method */
string::const_iterator FileDescriptor::write(const std::string& buffer,
                                             const bool write_all) {
  auto it = buffer.begin();

  do {
    it = write(it, buffer.end());
  } while (write_all and (it != buffer.end()));

  return it;
}

void FileDescriptor::set_blocking(const bool block) {
  int flags = SystemCall("fcntl F_GETFL", fcntl(fd_, F_GETFL));

  if (block) {
    flags = flags & ~O_NONBLOCK;
  } else {
    flags = flags | O_NONBLOCK;
  }

  SystemCall("fcntl F_SETFL", fcntl(fd_, F_SETFL, flags));
}
