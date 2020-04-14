#if !defined(__MATH)
#define __MATH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace math {
template <typename T> std::vector<T> flatten(const std::vector<std::vector<T>> &v) {
  std::vector<T> out;
  for (const auto &r : v)
    out.insert(out.end(), r.begin(), r.end());
  return out;
}

uint64_t ipow(int base, int exp) {
  uint64_t result = 1;
  for (;;) {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    if (!exp)
      break;
    base *= base;
  }

  return result;
}

template <typename T> T sum(const std::vector<T> &v) {
  return std::accumulate(std::begin(v), std::end(v), 0.0);
}

template <typename T> void transpose(std::vector<std::vector<T>> &v) {
  if (v.size() == 0)
    return;

  std::vector<std::vector<T>> trans_vec(v[0].size(), std::vector<T>());

  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v[i].size(); j++) {
      trans_vec[j].push_back(v[i][j]);
    }
  }

  v = trans_vec;
}

template <typename T> std::vector<T> sum2D(const std::vector<std::vector<T>> &v, int axis) {
  std::vector<std::vector<T>> arr(v);
  if (axis == 0) {
    transpose(arr);
  }
  std::vector<T> output(arr.size());

  for (size_t i = 0; i < output.size(); i++) {
    output[i] = sum(arr[i]);
  }
  return output;
}

template <typename T> double mean(const std::vector<T> &v) { return sum(v) / v.size(); }

template <typename T> double variance(const std::vector<T> &v) {
  double accum = 0.0;
  double mean_v = mean(v);

  for (size_t i = 0; i < v.size(); i++) {
    accum += pow((v[i] - mean_v), 2);
  }
  return accum / v.size();
}

template <typename T> double stdev(const std::vector<T> &v) { return std::sqrt(variance(v)); }

template <typename T> double moment(const std::vector<T> &v, uint m = 1) {
  if (m == 1) {
    return 0.0;
  }

  std::vector<double> mn(v.size());
  double mean_v = mean(v);

  for (size_t i = 0; i < v.size(); i++) {
    mn[i] = pow(v[i] - mean_v, m);
  }
  return mean(mn);
}

template <typename T> double kurtosis(const std::vector<T> &v, bool fisher = true) {
  double m2 = moment(v, 2);
  double m4 = moment(v, 4);

  double val = 0.0;

  if (m2 != 0) {
    val = m4 / pow(m2, 2);
  }

  if (fisher) {
    return val - 3;
  }

  return val;
}

template <typename T> double kurtosis(const std::vector<T> &v) {
  double m2 = moment(v, 2);
  double m4 = moment(v, 4);

  double val = 0.0;

  if (m2 != 0) {
    val = m4 / pow(m2, 2);
  }

  return val - 3; // fisher method
}

template <typename T> double skew(const std::vector<T> &v) {
  double m2 = moment(v, 2);
  double m3 = moment(v, 3);

  double val = 0.0;

  if (m2 != 0) {
    val = m3 / pow(m2, 1.5);
  }

  return val;
}

template <typename T>[[deprecated("Replaced by skew")]] double skewness(const std::vector<T> &v) {
  return skew(v);
}

template <typename T, typename A> T max(std::vector<T, A> const &vec) {
  return *std::max_element(vec.begin(), vec.end());
}

template <typename T, typename A> T min(std::vector<T, A> const &vec) {
  return *std::min_element(vec.begin(), vec.end());
}

template <typename T, typename A> uint argmax(std::vector<T, A> const &vec) {
  return static_cast<uint>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

template <typename T, typename A> uint argmin(std::vector<T, A> const &vec) {
  return static_cast<uint>(std::distance(vec.begin(), std::min_element(vec.begin(), vec.end())));
}

template <typename T> struct min_max_pair {
  min_max_pair(T min, T max) : min(min), max(max) {}
  T min;
  T max;
};

template <typename T> std::vector<double> normalize(std::vector<T> v) {
  std::vector<double> normalized(v.size(), 0.0);
  min_max_pair<T> mmp = getMinAndMax(v);

  double diff = mmp.max - mmp.min;
  for (unsigned int i = 0; i < v.size(); i++) {
    normalized[i] = (v[i] - mmp.min) / diff;
  }
  return normalized;
}

template <typename T> std::vector<T> arange(T start, T stop, T step = 1) {
  std::vector<T> values;
  for (T value = start; value < stop; value += step)
    values.push_back(value);
  return values;
}

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

double binaryEntropyFunction(double p) {
  // https://en.wikipedia.org/wiki/Binary_entropy_function
  if (p == 0 || p == 1) {
    return 0;
  }
  return -p * log2(p) - (1 - p) * log2(1 - p);
}

double constancyFunction(double p) { return 2 * abs(p - 0.5); }

template <typename T> std::vector<double> normalization(std::vector<T> data) {
  // https://www.statisticshowto.datasciencecentral.com/normalized/
  std::vector<double> output(data.size());
  double maxValue = (double)max(data);
  double minValue = (double)min(data);

  for (size_t i = 0; i < data.size(); i++) {
    output[i] = (data[i] - minValue) / (maxValue - minValue);
  }
  return output;
}

double distance(double a, double b) { return abs(a - b); }
double displacement(double a, double b) { return a - b; }

namespace random {
int randint(int min, int max) {
  std::random_device rd;  // only used once to initialise (seed) engine
  std::mt19937 rng(rd()); // random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int> uni(min, max); // guaranteed unbiased
  return uni(rng);
}

int randint(int max) { return randint(0, max); }

template <typename T> void shuffle(std::vector<T> &vec) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::shuffle(vec.begin(), vec.end(), mt);
}

} // namespace random

} // namespace math

#endif // __MATH
