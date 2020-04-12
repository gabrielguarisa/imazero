#if !defined(__MEMORY)
#define __MEMORY

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "../common/math.hpp"
#include "../common/utils.hpp"

class Memory {
private:
  std::vector<uint32_t> __mapping;
  uint __ndim;
  std::unordered_map<uint64_t, std::vector<uint32_t>> __memory;

  inline uint64_t __binaryToDecimal(const uint8_t *image) const {
    uint64_t index = 0;
    for (unsigned int i = 0; i < __mapping.size(); i++) {
      if (image[__mapping[i]] == 1) {
        index += math::ipow(2, i);
      }
    }
    return index;
  }

public:
  Memory(const std::vector<uint32_t> &mapping, const uint ndim)
      : __mapping(mapping), __ndim(ndim) {}
  ~Memory() {
    __mapping.clear();
    __memory.clear();
  }

  void write(const uint8_t *image) {
    utils::print(__binaryToDecimal(image));
  }
};

#endif // __MEMORY
