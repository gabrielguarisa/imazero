#if !defined(__MEMORY)
#define __MEMORY

#include <cstdint>
#include <unordered_map>
#include <utility>
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

  void write(const uint8_t *image, uint dim) {
    uint64_t addr = __binaryToDecimal(image);
    auto it = __memory.find(addr);
    if (it == __memory.end()) {
      it = __memory.insert(
          it, std::pair<uint64_t, std::vector<uint32_t>>(addr, std::vector<uint32_t>(__ndim, 0)));
    }

    it->second[dim]++;
  }

  std::vector<uint32_t> read(const uint8_t *image) {
    uint64_t addr = __binaryToDecimal(image);
    auto it = __memory.find(addr);
    if (it == __memory.end()) {
      return std::vector<uint32_t>(__ndim, 0);
    }
    return it->second;
  }
};

#endif // __MEMORY
