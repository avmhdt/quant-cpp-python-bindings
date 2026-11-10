// Arrow C Data Interface importer: hand-built ArrowSchema/ArrowArray structs.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qkern/arrow_import.hpp"
#include "qkern/rolling_moments.hpp"

namespace {

void noop_schema(ArrowSchema* /*unused*/) {}
void noop_array(ArrowArray* /*unused*/) {}

struct Fixture {
  ArrowSchema schema{};
  ArrowArray array{};
  std::array<const void*, 2> buffers{};

  Fixture(const char* format, const void* values, std::int64_t length, std::int64_t offset,
          const std::uint8_t* validity, std::int64_t null_count) {
    schema.format = format;
    schema.release = &noop_schema;
    buffers = {validity, values};
    array.length = length;
    array.offset = offset;
    array.null_count = null_count;
    array.n_buffers = 2;
    array.buffers = buffers.data();
    array.release = &noop_array;
  }
};

TEST(ArrowImport, Float64WithOffsetAndNulls) {
  const std::array<double, 6> values{9.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  // offset 1: logical elements are values[1..5]; element 2 (value 3.0) is null.
  // Bits for offset 1 + i: set for i = 0, 1, 3, 4 -> bits 1, 2, 4, 5.
  const std::array<std::uint8_t, 1> validity{0b00110110U};
  Fixture f("g", values.data(), 5, 1, validity.data(), 1);
  const qkern::ArrowValues v = qkern::import_arrow(&f.schema, &f.array);
  EXPECT_EQ(v.kind, qkern::ArrowKind::Float64);
  const auto view = v.view<double>();
  EXPECT_EQ(view[0], 1.0);
  EXPECT_EQ(view[1], 2.0);
  EXPECT_TRUE(std::isnan(view[2]));
  EXPECT_EQ(view[4], 5.0);
  std::vector<double> out(5);
  qkern::rolling_mean(view, qkern::OutView{out.data(), 5, 1}, qkern::RollingConfig{2, 1});
  EXPECT_EQ(out[3], 4.0);  // the null left the window's valid set
}

TEST(ArrowImport, Float32WithoutNullsHasNoBitmap) {
  const std::array<float, 3> values{1.0F, 2.0F, 3.0F};
  const std::array<std::uint8_t, 1> validity{0xFFU};
  Fixture f("f", values.data(), 3, 0, validity.data(), 0);
  const qkern::ArrowValues v = qkern::import_arrow(&f.schema, &f.array);
  EXPECT_EQ(v.kind, qkern::ArrowKind::Float32);
  EXPECT_EQ(v.validity, nullptr);  // null_count == 0: read the values buffer in place
  EXPECT_EQ(v.view<float>()[2], 3.0);
}

TEST(ArrowImport, RejectsUnsupportedArrays) {
  const std::array<double, 2> values{1.0, 2.0};
  Fixture wrong_format("l", values.data(), 2, 0, nullptr, 0);
  EXPECT_THROW((void)qkern::import_arrow(&wrong_format.schema, &wrong_format.array),
               std::invalid_argument);
  Fixture released("g", values.data(), 2, 0, nullptr, 0);
  released.array.release = nullptr;
  EXPECT_THROW((void)qkern::import_arrow(&released.schema, &released.array), std::invalid_argument);
  Fixture negative("g", values.data(), 2, -1, nullptr, 0);
  EXPECT_THROW((void)qkern::import_arrow(&negative.schema, &negative.array), std::invalid_argument);
  Fixture no_values("g", nullptr, 2, 0, nullptr, 0);
  EXPECT_THROW((void)qkern::import_arrow(&no_values.schema, &no_values.array),
               std::invalid_argument);
  Fixture nested("g", values.data(), 2, 0, nullptr, 0);
  nested.array.n_children = 1;
  EXPECT_THROW((void)qkern::import_arrow(&nested.schema, &nested.array), std::invalid_argument);
  EXPECT_THROW((void)qkern::import_arrow(nullptr, &nested.array), std::invalid_argument);
  Fixture empty("g", nullptr, 0, 0, nullptr, 0);
  EXPECT_EQ(qkern::import_arrow(&empty.schema, &empty.array).length, 0U);
}

}  // namespace
