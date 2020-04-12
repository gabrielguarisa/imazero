#if !defined(__UITLS)
#define __UITLS

#include <iostream>
#include <map>
#include <vector>

namespace utils {
template <typename K, typename V> K findByValue(K &key, std::map<K, V> data, V value) {
  for (auto &it : data) {
    if (it->second == value) {
      key = it->first;
      return true;
    }
  }
  return false;
}

template <typename T> std::map<T, std::vector<uint>> groupByValue(std::vector<T> data) {
  std::map<T, std::vector<uint>> bins;
  for (uint i = 0; i < data.size(); i++) {
    auto it = bins.find(data[i]);
    if (it == bins.end()) {
      bins.insert(it, std::pair<T, std::vector<uint>>(data[i], {i}));
    } else {
      it->second.push_back(i);
    }
  }
  return bins;
}

template <typename T> std::vector<T> arange(T start, T stop, T step) {
  std::vector<T> values;
  for (T value = start; value < stop; value += step)
    values.push_back(value);
  return values;
}

template <typename T> std::vector<T> arange(T start, T stop) { return arange<T>(start, stop, 1); }

template <typename T> std::vector<T> arange(T stop) { return arange<T>(0, stop, 1); }

template <typename T> std::vector<T> ranges(T start, T stop, size_t num) {
  std::vector<T> values;
  T step = (stop - start) / num;
  T value = start;
  for (size_t i = 0; i < num; i++) {
    values.push_back(value);
    value += step;
  }

  return values;
}

template <typename T> std::vector<T> ranges(T stop, size_t num) { return ranges(0, stop, num); }

template <typename T> void print(T value) { std::cout << value << std::endl; }

template <typename T, typename... Args> void print(T value, Args... args) {
  std::cout << value << " ";
  print(args...);
}
} // namespace utils

#endif // __UITLS
