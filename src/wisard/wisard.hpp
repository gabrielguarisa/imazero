#if !defined(__WISARD)
#define __WISARD

#include "../common/utils.hpp"
#include "bleaching.hpp"
#include "memory.hpp"
#include "stochastic.hpp"
#include <cstdint>
#include <vector>

class WiSARD {
private:
  std::vector<std::vector<uint32_t>> __mapping;
  std::vector<Memory> __memories;
  uint __ndim;
  uint __entrySize;
  Bleaching __bleaching;

public:
  WiSARD(const std::vector<std::vector<uint32_t>> &mapping, uint ndim, uint entrySize)
      : __mapping(mapping), __ndim(ndim), __entrySize(entrySize) {
    __memories = std::vector<Memory>(mapping.size(), Memory(ndim));
    // __bleaching.setConfidenceThreshold(0.1);
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
      std::vector<double> prob = __bleaching.run(votes[i]);
      output[i] = math::argmax(prob);

      // std::vector<uint32_t> sum_values = math::sum2D(votes[i], 0);
      // output[i] = math::argmax(sum_values);
    }
    return output;
  }

  std::vector<std::vector<double>> azharMeasures(const uint8_t *X, const uint8_t *y, uint lenght) {
    std::vector<std::vector<std::vector<uint32_t>>> votes = getVotes(X, lenght);
    std::vector<std::vector<double>> output(__memories.size(), std::vector<double>(3, 0.0));
    double sumValue = 1.0 / (double)lenght;

    std::vector<uint32_t> states;
    for (size_t i = 0; i < votes.size(); i++) {
      states = azharStates(votes[i], y[i]);

      for (size_t j = 0; j < states.size(); j++) {
        output[j][states[j]] += sumValue;
      }
    }

    return output;
  }

  std::vector<std::vector<double>> guarisaMeasures(const uint8_t *X, const uint8_t *y, uint lenght) {
    std::vector<std::vector<std::vector<uint32_t>>> votes = getVotes(X, lenght);
    std::vector<std::vector<double>> output(__memories.size(), std::vector<double>(4, 0.0));
    double sumValue = 1.0 / (double)lenght;

    std::vector<uint32_t> states;
    for (size_t i = 0; i < votes.size(); i++) {
      states = guarisaStates(votes[i], y[i]);

      for (size_t j = 0; j < states.size(); j++) {
        output[j][states[j]] += sumValue;
      }
    }

    return output;
  }
};

#endif // __WISARD
