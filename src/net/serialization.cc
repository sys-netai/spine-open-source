#include "serialization.hh"

using namespace std;

string put_field( const uint16_t n){
  const uint16_t network_order = htobe16(n);
  return string(reinterpret_cast<const char*>(&network_order), sizeof(network_order));
}

uint16_t get_uint16(const char * data)
{
  return be16toh(*reinterpret_cast<const uint16_t *>(data));
}
