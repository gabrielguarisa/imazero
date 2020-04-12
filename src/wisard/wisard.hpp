#if !defined(__WISARD)
#define __WISARD

#include "../common/utils.hpp"
#include <cstdint>
#include <vector>

class WiSARD {
private:
  std::vector<std::vector<uint32_t>> __mapping;
  uint __ndim;
  uint __entrySize;

public:
  WiSARD(const std::vector<std::vector<uint32_t>> &mapping, uint ndim, uint entrySize)
      : __mapping(mapping), __ndim(ndim), __entrySize(entrySize) {
    for (size_t i = 0; i < mapping.size(); i++) {
      for (size_t j = 0; j < mapping[i].size(); j++) {
        std::cout << mapping[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }
  ~WiSARD() { __mapping.clear(); }

  void train(const uint8_t *X, const uint8_t *y, uint lenght) {
    for (size_t i = 0; i < lenght; i++) {
      std::cout << (int)y[i]<< ", ";
    }
  }
};

#endif // __WISARD
