// save()/load() round trips (pickling support, FR-09): a restored object continues
// bit-identically, and malformed states are rejected.
#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qkern/ewm.hpp"
#include "qkern/rolling_minmax.hpp"
#include "qkern/rolling_moments.hpp"
#include "qkern/rolling_ols.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::RollingConfig;
using qkern_test::kInf;

std::vector<double> data(unsigned seed) {
  auto x = qkern_test::gaussian(3000, seed);
  qkern_test::add_nans(x, 0.05, seed + 1);
  x[500] = kInf;
  x[1700] = -kInf;
  return x;
}

// Push the first half into `a`, copy its state into `b`, then feed both the second half.
template <class T, class Push, class Read>
void check_continuation(T a, T b, const std::vector<double>& x, Push push, Read read) {
  const std::size_t half = x.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    push(a, i);
  }
  b.load(a.save());
  for (std::size_t i = half; i < x.size(); ++i) {
    push(a, i);
    push(b, i);
    qkern_test::expect_identical({read(a)}, {read(b)});
  }
}

TEST(State, RollingMomentsContinuesBitIdentically) {
  const auto x = data(91);
  const RollingConfig cfg{40, 5};
  check_continuation(
      qkern::RollingMoments(cfg), qkern::RollingMoments(cfg), x,
      [&](qkern::RollingMoments& s, std::size_t i) { s.push(x[i]); },
      [](const qkern::RollingMoments& s) { return s.var() + s.mean(); });
}

TEST(State, EwmContinuesBitIdentically) {
  const auto x = data(92);
  const qkern::EwmConfig cfg{0.07, false, 3};
  check_continuation(
      qkern::Ewm(cfg), qkern::Ewm(cfg), x, [&](qkern::Ewm& s, std::size_t i) { s.push(x[i]); },
      [](const qkern::Ewm& s) { return s.var() + s.mean(); });
}

TEST(State, RollingMinMaxContinuesBitIdentically) {
  const auto x = data(93);
  const RollingConfig cfg{25, 1};
  check_continuation(
      qkern::RollingMinMax(cfg), qkern::RollingMinMax(cfg), x,
      [&](qkern::RollingMinMax& s, std::size_t i) { s.push(x[i]); },
      [](const qkern::RollingMinMax& s) { return s.max() - s.min(); });
}

TEST(State, RollingOlsContinuesBitIdentically) {
  const auto x = data(94);
  const auto y = data(95);
  const RollingConfig cfg{30, 3};
  check_continuation(
      qkern::RollingOls(cfg), qkern::RollingOls(cfg), x,
      [&](qkern::RollingOls& s, std::size_t i) { s.push(y[i], x[i]); },
      [](const qkern::RollingOls& s) { return s.beta() + s.alpha(); });
}

TEST(State, MalformedStatesAreRejected) {
  qkern::RollingMoments m(RollingConfig{3, 1});
  qkern::State st = m.save();
  st.ints.pop_back();
  EXPECT_THROW(m.load(st), std::invalid_argument);
  qkern::State big;
  big.buffer.assign(4, 1.0);  // longer than the window
  big.ints = {4, 0, 0, 0, 0, 0};
  big.reals.assign(6, 0.0);
  EXPECT_THROW(m.load(big), std::invalid_argument);

  qkern::Ewm e(qkern::EwmConfig{0.5});
  qkern::State es = e.save();
  es.ints[1] = 2;
  EXPECT_THROW(e.load(es), std::invalid_argument);

  qkern::RollingMinMax mm(RollingConfig{3, 1}, false, true);
  mm.push(1.0);
  qkern::State ms = mm.save();
  ms.ints[4] = 1;  // claims a min deque this object does not track
  ms.ints.push_back(0);
  ms.reals.push_back(1.0);
  EXPECT_THROW(mm.load(ms), std::invalid_argument);
  EXPECT_TRUE(mm.tracks_max());
  EXPECT_FALSE(mm.tracks_min());

  qkern::RollingOls o(RollingConfig{3, 1});
  qkern::State os = o.save();
  os.buffer.push_back(1.0);  // odd length
  EXPECT_THROW(o.load(os), std::invalid_argument);
}

}  // namespace
