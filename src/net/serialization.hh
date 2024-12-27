#ifndef SERIALIZATION_HH
#define SERIALIZATION_HH

#include <string>
#include <cstdint>

std::string put_field(const uint16_t n);
uint16_t get_uint16(const char * data);

#endif /* SERIALIZATION_HH */