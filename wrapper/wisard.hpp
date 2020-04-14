#if !defined(__WISARD_WRAPPER)
#define __WISARD_WRAPPER

#include "../src/common/math.hpp"
#include "../src/wisard/wisard.hpp"
#include <cstdint>
#include <vector>

#define WISARD_SELF (static_cast<WiSARD *>(self))

extern "C" {

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

void wisard_classify(void *self, void *const X, void *const output, uint lenght) {
  std::vector<uint32_t> votes = WISARD_SELF->classify((uint8_t *)X, lenght);
  std::copy(votes.begin(), votes.end(), (uint32_t *)output);
}

void wisard_azhar_measures(void *self, void *const X, void *const y, void *const output, uint lenght) {
  std::vector<double> measures =
      math::flatten(WISARD_SELF->azharMeasures((uint8_t *)X, (uint8_t *)y, lenght));
  std::copy(measures.begin(), measures.end(), (double *)output);
}
}
#endif // __WISARD_WRAPPER
