#if !defined(__WISARD)
#define __WISARD

#include "../common/utils.hpp"
#include "memory.hpp"
#include <cstdint>
#include <vector>

class WiSARD {
private:
  std::vector<std::vector<uint32_t>> __mapping;
  std::vector<Memory> __memories;
  uint __ndim;
  uint __entrySize;

public:
  WiSARD(const std::vector<std::vector<uint32_t>> &mapping, uint ndim, uint entrySize)
      : __mapping(mapping), __ndim(ndim), __entrySize(entrySize) {
    __memories = std::vector<Memory>(mapping.size(), Memory(ndim));
  }
  ~WiSARD() {
    __mapping.clear();
    __memories.clear();
  }

  void train(const uint8_t *X, const uint8_t *y, uint lenght) {
    for (size_t i = 0; i < lenght; i++) {
      for (size_t j = 0; j < __memories.size(); j++) {
        __memories[j].write(&X[i * __entrySize], y[i], __mapping[j]);
      }
    }
  }

  std::vector<std::vector<std::vector<uint32_t>>> getVotes(const uint8_t *X, uint lenght) {
    std::vector<std::vector<std::vector<uint32_t>>> votes(
        lenght, std::vector<std::vector<uint32_t>>(__memories.size()));
    for (size_t i = 0; i < lenght; i++) {
      for (size_t j = 0; j < __memories.size(); j++) {
        votes[i][j] = __memories[j].read(&X[i * __entrySize], __mapping[j]);
      }
    }
    return votes;
  }

  std::vector<uint32_t> classify(const uint8_t *X, uint lenght) {
    std::vector<uint32_t> output(lenght);
    std::vector<std::vector<std::vector<uint32_t>>> votes = getVotes(X, lenght);

    for (size_t i = 0; i < votes.size(); i++) {
      std::vector<uint32_t> sum_values = math::sum2D(votes[i], 0);
      output[i] = math::argmax(sum_values);
    }
    return output;
  }
};

#endif // __WISARD
