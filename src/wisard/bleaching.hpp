#if !defined(__BLEACHING)
#define __BLEACHING

#include "../common/math.hpp"
#include <iostream>
#include <vector>

class Bleaching {
private:
  double __confidenceThreshold;

  double __calcConfidence(const std::vector<double> &counters) {
    double maxValue = 0.0;
    double secondMax = 0.0;
    bool tie = true;

    for (size_t i = 0; i < counters.size(); i++) {
      if (counters[i] > maxValue) {
        secondMax = maxValue;
        maxValue = counters[i];
        tie = false;
      } else if (counters[i] > secondMax) {
        secondMax = counters[i];
      } else if (counters[i] == maxValue) {
        tie = true;
      }
    }

    if (maxValue == 0 || tie) {
      return 0.0;
    }

    return 1 - secondMax / maxValue;
  }

public:
  Bleaching() {}

  std::vector<double> run(const std::vector<std::vector<uint32_t>> &votes) {
    uint b = 1;
    double confidence = 0.0;
    std::vector<double> counters;
    std::vector<double> prevCounters;
    double resultSum = 1;

    while (confidence < __confidenceThreshold) {
      counters = std::vector<double>(votes[0].size(), 0.0);
      for (size_t i = 0; i < votes[0].size(); i++) {
        for (size_t j = 0; j < votes.size(); j++) {
          if (votes[j][i] >= b) {
            counters[i] += 1;
          }
        }
      }
      confidence = __calcConfidence(counters);

      resultSum = math::sum(counters);

      if (resultSum == 0) {
        counters = prevCounters;
        break;
      }
      b++;
      prevCounters = counters;
    }

    if (resultSum > 1) {
      for (size_t i = 0; i < counters.size(); i++) {
        counters[i] = counters[i] / resultSum;
      }
    }

    return counters;
  }

  void setConfidenceThreshold(double confidenceThreshold) {
    __confidenceThreshold = confidenceThreshold;
  }
};

#endif // __BLEACHING
