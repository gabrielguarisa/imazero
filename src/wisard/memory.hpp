#if !defined(__MEMORY)
#define __MEMORY

#include <cstdint>
#include <iostream>
#include <vector>

class Memory {
private:
  std::vector<uint32_t> __mapping;

public:
  Memory(const std::vector<uint32_t> &mapping) : __mapping(mapping) {
    for (size_t i = 0; i < __mapping.size(); i++) {
      std::cout << __mapping[i] << " ";
    }
    std::cout << std::endl;
  }
  ~Memory() { __mapping.clear(); }
};

#endif // __MEMORY
