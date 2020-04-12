#if !defined(__WISARD_WRAPPER)
#define __WISARD_WRAPPER

#include "../src/wisard/memory.hpp"
#include <cstdint>
#include <vector>

#define MEMORY_SELF (static_cast<Memory *>(self))

extern "C" {

void *memory_create(void *const _ptr, const int length) {
  uint32_t *const ptr = (uint32_t *)_ptr;
  std::vector<uint32_t> mapping(ptr, ptr + length);
  return static_cast<void *>(new Memory(mapping));
}

void memory_destroy(void *self) { delete MEMORY_SELF; }
}

#endif // __WISARD_WRAPPER
