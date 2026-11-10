#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include "qkern/nan_policy.hpp"
#include "qkern/ring_buffer.hpp"
#include "qkern/version.hpp"
#include "qkern/views.hpp"

namespace {

TEST(StridedView, ContiguousPositiveAndNegativeStride) {
  const std::array<double, 5> buf{0.0, 1.0, 2.0, 3.0, 4.0};
  const qkern::StridedView<double> fwd{buf.data(), 5, 1};
  EXPECT_EQ(fwd[0], 0.0);
  EXPECT_EQ(fwd[4], 4.0);

  const qkern::StridedView<double> every2{buf.data(), 3, 2};
  EXPECT_EQ(every2[2], 4.0);

  const qkern::StridedView<double> rev{buf.data() + 4, 5, -1};
  EXPECT_EQ(rev[0], 4.0);
  EXPECT_EQ(rev[4], 0.0);
}

TEST(StridedView, FloatIsWidenedToDouble) {
  const std::array<float, 2> buf{0.1F, 2.5F};
  const qkern::StridedView<float> v{buf.data(), 2, 1};
  EXPECT_EQ(v[0], static_cast<double>(0.1F));
  EXPECT_EQ(v[1], 2.5);
}

TEST(StridedView, ValidityBitmapReadsNullAsNaN) {
  const std::array<double, 10> buf{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // Bits (LSB first) for offset 1: elements 0..9 map to bits 1..10.
  // Clear element 2 (bit 3) and element 9 (bit 10).
  const std::array<std::uint8_t, 2> bitmap{0b11110111U, 0b00000011U};
  const qkern::StridedView<double> v{buf.data(), 10, 1, bitmap.data(), 1};
  EXPECT_EQ(v[0], 0.0);
  EXPECT_EQ(v[1], 1.0);
  EXPECT_TRUE(std::isnan(v[2]));
  EXPECT_EQ(v[8], 8.0);
  EXPECT_TRUE(std::isnan(v[9]));
}

TEST(OutView, WritesThroughStride) {
  std::array<double, 6> buf{};
  const qkern::OutView out{buf.data() + 5, 3, -2};
  out[0] = 1.0;
  out[1] = 2.0;
  out[2] = 3.0;
  EXPECT_EQ(buf[5], 1.0);
  EXPECT_EQ(buf[3], 2.0);
  EXPECT_EQ(buf[1], 3.0);
}

TEST(RingBuffer, WrapsAround) {
  qkern::RingBuffer rb(3);
  EXPECT_TRUE(rb.empty());
  rb.push_back(1.0);
  rb.push_back(2.0);
  rb.push_back(3.0);
  EXPECT_TRUE(rb.full());
  EXPECT_EQ(rb.pop_front(), 1.0);
  rb.push_back(4.0);
  EXPECT_EQ(rb[0], 2.0);
  EXPECT_EQ(rb[1], 3.0);
  EXPECT_EQ(rb[2], 4.0);
  EXPECT_EQ(rb.front(), 2.0);
  rb.clear();
  EXPECT_EQ(rb.size(), 0U);
  EXPECT_EQ(rb.capacity(), 3U);
}

TEST(Config, Validation) {
  EXPECT_THROW(qkern::validate(qkern::RollingConfig{0, 1}), std::invalid_argument);
  EXPECT_THROW(qkern::validate(qkern::RollingConfig{3, 0}), std::invalid_argument);
  EXPECT_THROW(qkern::validate(qkern::RollingConfig{3, 4}), std::invalid_argument);
  EXPECT_THROW(qkern::validate(qkern::RollingConfig{3, 3, -1}), std::invalid_argument);
  EXPECT_NO_THROW(qkern::validate(qkern::RollingConfig{3, 3}));
}

TEST(Config, ParseNanPolicy) {
  EXPECT_EQ(qkern::parse_nan_policy("skip"), qkern::NanPolicy::Skip);
  EXPECT_EQ(qkern::parse_nan_policy("propagate"), qkern::NanPolicy::Propagate);
  EXPECT_EQ(qkern::parse_nan_policy("raise"), qkern::NanPolicy::Raise);
  EXPECT_THROW((void)qkern::parse_nan_policy("ignore"), std::invalid_argument);
}

TEST(Version, MacrosAreDefined) {
  EXPECT_GE(QKERN_VERSION_MAJOR, 0);
  EXPECT_GT(sizeof(QKERN_VERSION_STRING), 1U);
}

}  // namespace
