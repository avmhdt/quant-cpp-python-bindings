// Fixed-capacity ring buffer, sized once at construction (no allocation in the hot loop).
#pragma once

#include <cstddef>
#include <vector>

namespace qkern {

class RingBuffer {
 public:
  explicit RingBuffer(std::size_t capacity) : buf_(capacity == 0 ? 1 : capacity) {}

  [[nodiscard]] std::size_t capacity() const noexcept { return buf_.size(); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool full() const noexcept { return size_ == buf_.size(); }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  // Oldest element first: (*this)[0] is the front.
  [[nodiscard]] double operator[](std::size_t i) const noexcept { return buf_[wrap(head_ + i)]; }
  [[nodiscard]] double front() const noexcept { return buf_[head_]; }

  // Precondition: !full().
  void push_back(double x) noexcept {
    buf_[wrap(head_ + size_)] = x;
    ++size_;
  }

  // Precondition: !empty().
  double pop_front() noexcept {
    const double x = buf_[head_];
    head_ = wrap(head_ + 1);
    --size_;
    return x;
  }

  void clear() noexcept {
    head_ = 0;
    size_ = 0;
  }

 private:
  [[nodiscard]] std::size_t wrap(std::size_t i) const noexcept {
    return i >= buf_.size() ? i - buf_.size() : i;
  }

  std::vector<double> buf_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

}  // namespace qkern
