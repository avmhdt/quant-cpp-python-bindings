// Exponentially weighted mean, variance and standard deviation (adjust=False,
// ignore_na=True). The recurrence is a port of pandas' ewmcov (pandas/_libs/window/
// aggregations.pyx) with x = y, so the mean equals pandas' ewm(...).mean() update and the
// variance pandas' ewm(...).var(bias) update:
//
//   old_wt <- (1 - alpha) * old_wt          (old_wt is reset to 1 after every observation)
//   mean   <- (old_wt * mean + alpha * x) / (old_wt + alpha)        unless mean == x
//   cov    <- (old_wt * (cov + (mean_old - mean)^2) + alpha * (x - mean)^2) / (old_wt + alpha)
//   sum_wt, sum_wt2 track the weights for the unbiased correction
//   var    =  cov * sum_wt^2 / (sum_wt^2 - sum_wt2)                 (bias=False)
//
// Semantics:
//   * NaN input with NanPolicy::Skip leaves the state unchanged; the output is the current
//     value when count() >= min_periods. NanPolicy::Propagate: the window is unbounded, so the
//     first NaN makes every later output NaN until reset(). NanPolicy::Raise is checked by
//     the batch drivers.
//   * Infinities follow the recurrence: the mean becomes +-inf (NaN once both signs were
//     seen) and the variance NaN from then on, matching pandas.
#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

#include "qkern/detail.hpp"
#include "qkern/nan_policy.hpp"
#include "qkern/views.hpp"

namespace qkern {

// alpha = 2 / (span + 1), span >= 1.
[[nodiscard]] inline double alpha_from_span(double span) {
  if (!(span >= 1.0) || std::isinf(span)) {
    throw std::invalid_argument("span must satisfy span >= 1");
  }
  return 2.0 / (span + 1.0);
}

// alpha = 1 - exp(-ln 2 / halflife), halflife > 0.
[[nodiscard]] inline double alpha_from_halflife(double halflife) {
  if (!(halflife > 0.0) || std::isinf(halflife)) {
    throw std::invalid_argument("halflife must be > 0");
  }
  return 1.0 - std::exp(-std::numbers::ln2 / halflife);
}

class Ewm {
 public:
  explicit Ewm(const EwmConfig& cfg) : cfg_((validate(cfg), cfg)), decay_(1.0 - cfg.alpha) {}

  void push(double x) noexcept {
    if (std::isnan(x)) {
      if (cfg_.nan == NanPolicy::Propagate) {
        poisoned_ = true;
      }
      return;
    }
    ++nobs_;
    if (nobs_ == 1) {
      mean_ = x;
      return;
    }
    const double new_wt = cfg_.alpha;
    sum_wt_ *= decay_;
    sum_wt2_ *= decay_ * decay_;
    old_wt_ *= decay_;
    const double mean_old = mean_;
    if (mean_ != x) {
      mean_ = ((old_wt_ * mean_old) + (new_wt * x)) / (old_wt_ + new_wt);
    }
    cov_ = ((old_wt_ * (cov_ + ((mean_old - mean_) * (mean_old - mean_)))) +
            (new_wt * ((x - mean_) * (x - mean_)))) /
           (old_wt_ + new_wt);
    sum_wt_ += new_wt;
    sum_wt2_ += new_wt * new_wt;
    old_wt_ += new_wt;
    // adjust=False: renormalise so the previous weight is 1 again.
    sum_wt_ /= old_wt_;
    sum_wt2_ /= old_wt_ * old_wt_;
    old_wt_ = 1.0;
  }

  // Valid (non-NaN) observations pushed since construction or reset().
  [[nodiscard]] std::size_t count() const noexcept { return nobs_; }

  [[nodiscard]] bool ready() const noexcept {
    const std::size_t min_obs = cfg_.min_periods > 0 ? cfg_.min_periods : 1;
    return !poisoned_ && nobs_ >= min_obs;
  }

  [[nodiscard]] double mean() const noexcept { return ready() ? mean_ : kNaN; }

  [[nodiscard]] double var() const noexcept {
    if (!ready()) {
      return kNaN;
    }
    if (cfg_.bias) {
      return cov_;
    }
    const double numerator = sum_wt_ * sum_wt_;
    const double denominator = numerator - sum_wt2_;
    return denominator > 0.0 ? (numerator / denominator) * cov_ : kNaN;
  }

  [[nodiscard]] double stddev() const noexcept { return std::sqrt(var()); }

  void reset() noexcept {
    nobs_ = 0;
    poisoned_ = false;
    mean_ = cov_ = 0.0;
    sum_wt_ = sum_wt2_ = old_wt_ = 1.0;
  }

  [[nodiscard]] const EwmConfig& config() const noexcept { return cfg_; }

  [[nodiscard]] State save() const {
    State st;
    st.ints = {nobs_, poisoned_ ? 1U : 0U};
    st.reals = {mean_, cov_, sum_wt_, sum_wt2_, old_wt_};
    return st;
  }

  void load(const State& st) {
    detail::check_state(st.buffer.empty() && st.ints.size() == 2 && st.reals.size() == 5 &&
                        st.ints[1] <= 1);
    nobs_ = static_cast<std::size_t>(st.ints[0]);
    poisoned_ = st.ints[1] == 1;
    mean_ = st.reals[0];
    cov_ = st.reals[1];
    sum_wt_ = st.reals[2];
    sum_wt2_ = st.reals[3];
    old_wt_ = st.reals[4];
  }

 private:
  EwmConfig cfg_;
  double decay_;
  std::size_t nobs_ = 0;
  bool poisoned_ = false;
  double mean_ = 0.0;
  double cov_ = 0.0;
  double sum_wt_ = 1.0;
  double sum_wt2_ = 1.0;
  double old_wt_ = 1.0;
};

template <Real T>
void ewm_mean(StridedView<T> x, OutView out, const EwmConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  Ewm state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.mean();
  }
}

template <Real T>
void ewm_var(StridedView<T> x, OutView out, const EwmConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  Ewm state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.var();
  }
}

template <Real T>
void ewm_std(StridedView<T> x, OutView out, const EwmConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  Ewm state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.stddev();
  }
}

}  // namespace qkern
