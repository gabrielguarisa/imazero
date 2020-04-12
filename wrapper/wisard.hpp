#if !defined(__WISARD_WRAPPER)
#define __WISARD_WRAPPER

#include "../src/wisard/memory.hpp"
#include <cstdint>
#include <vector>

#define MEMORY_SELF (static_cast<Memory *>(self))

extern "C" {

void *memory_create(void *const __mapping, const int length, const uint ndim) {
  uint32_t *const _mapping = (uint32_t *)__mapping;
  std::vector<uint32_t> mapping(_mapping, _mapping + length);
  return static_cast<void *>(new Memory(mapping, ndim));
}

void memory_destroy(void *self) { delete MEMORY_SELF; }

void memory_write(void *self, void *const image, uint dim) { MEMORY_SELF->write((uint8_t *)image, dim); }
}
#endif // __WISARD_WRAPPER
