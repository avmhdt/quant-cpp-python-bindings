// Minimal qkern::core consumer: rolling mean/std and a streaming EWM over a short series.
#include <cmath>
#include <cstdio>
#include <vector>

#include "qkern/qkern.hpp"

int main() {
  const std::vector<double> x{1.0, 2.0, 4.0, 8.0, 16.0};
  std::vector<double> mean(x.size());
  std::vector<double> sd(x.size());
  const qkern::RollingConfig cfg{3, 2};
  qkern::rolling_mean(qkern::StridedView<double>{x.data(), x.size(), 1},
                      qkern::OutView{mean.data(), mean.size(), 1}, cfg);
  qkern::rolling_std(qkern::StridedView<double>{x.data(), x.size(), 1},
                     qkern::OutView{sd.data(), sd.size(), 1}, cfg);

  qkern::Ewm ewm(qkern::EwmConfig{qkern::alpha_from_span(3.0)});
  for (double v : x) {
    ewm.push(v);
  }

  std::printf("qkern %s\n", QKERN_VERSION_STRING);
  for (std::size_t i = 0; i < x.size(); ++i) {
    std::printf("x=%5.1f  mean=%8.4f  std=%8.4f\n", x[i], mean[i], sd[i]);
  }
  std::printf("ewm mean=%.6f\n", ewm.mean());
  // Window {4, 8, 16}: mean 28/3.
  return std::fabs(mean[4] - 28.0 / 3.0) < 1e-12 ? 0 : 1;
}
