#if !defined(__WISARD_WRAPPER)
#define __WISARD_WRAPPER

#include "../src/wisard/memory.hpp"
#include "../src/wisard/wisard.hpp"
#include <cstdint>
#include <vector>

#define MEMORY_SELF (static_cast<Memory *>(self))
#define WISARD_SELF (static_cast<WiSARD *>(self))

extern "C" {

// Memory
void *memory_create(void *const __mapping, const int length, const uint ndim) {
  uint32_t *const _mapping = (uint32_t *)__mapping;
  std::vector<uint32_t> mapping(_mapping, _mapping + length);
  return static_cast<void *>(new Memory(mapping, ndim));
}

void memory_destroy(void *self) { delete MEMORY_SELF; }

void memory_write(void *self, void *const image, uint dim) {
  MEMORY_SELF->write((uint8_t *)image, dim);
}

void memory_read(void *self, void *const image, void *const output) {
  std::vector<uint32_t> result = MEMORY_SELF->read((uint8_t *)image);
  std::copy(result.begin(), result.end(), (uint32_t *)output);
}

// WiSARD
void *wisard_create(void *const __mapping, const uint rows, const uint cols, const uint ndim,
                    const uint entrySize) {
  uint32_t *const _mapping = (uint32_t *)__mapping;
  std::vector<std::vector<uint32_t>> mapping(rows);

  for (size_t i = 0; i < rows; i++) {
    mapping[i] = std::vector<uint32_t>(_mapping + (i * cols), _mapping + (i * cols) + cols);
  }
  return static_cast<void *>(new WiSARD(mapping, ndim, entrySize));
}

void wisard_destroy(void *self) { delete WISARD_SELF; }

void wisard_train(void *self, void *const X, void *const y, uint lenght) {
  WISARD_SELF->train((uint8_t *)X, (uint8_t *)y, lenght);
}
}
#endif // __WISARD_WRAPPER
