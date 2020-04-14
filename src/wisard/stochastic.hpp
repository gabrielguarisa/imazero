#if !defined(__STOCHASTIC)
#define __STOCHASTIC

#include <cstdint>
#include <vector>

uint32_t tupleAzharStates(const std::vector<uint32_t> &counters,
                          uint32_t trueLabel) { // tuple response state to a single pattern
  uint32_t maxValue = 0;
  bool tie = true;

  for (size_t i = 0; i < counters.size(); i++) {
    if (counters[i] > maxValue) {
      maxValue = counters[i];
      tie = false;
    } else if (counters[i] == maxValue) {
      tie = true;
    }
  }

  if (maxValue == 0 || tie) {
    return 1; // neutral
  } else if (counters[trueLabel] == maxValue) {
    return 2; // good
  }
  return 0; // bad
}

std::vector<uint32_t> azharStates(const std::vector<std::vector<uint32_t>> &counters,
                                  uint32_t trueLabel) { // all tuples states to a single pattern
  std::vector<uint32_t> stateCounter(counters.size());

  for (size_t i = 0; i < counters.size(); i++) {
    stateCounter[i] = tupleAzharStates(counters[i], trueLabel);
  }
  return stateCounter;
}

#endif // __STOCHASTIC
