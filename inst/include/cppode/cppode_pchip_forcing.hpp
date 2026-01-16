/*
 * PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) forcing functions
 *
 * This header provides:
 * - PchipForcing<T>: Time-dependent forcing with PCHIP interpolation
 * - Fritsch-Carlson monotonicity-preserving slope computation
 * - C1 continuous interpolation with analytical derivatives
 * - Support for FADBAD++ automatic differentiation types (F<T>, F<F<T>>)
 * - Zero extrapolation outside defined time range (safe default)
 *
 * Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_PCHIP_FORCING_HPP
#define CPPODE_PCHIP_FORCING_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cppode {

/**
 * PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) forcing function.
 *
 * Computes cubic spline coefficients from raw (time, value) data using the
 * Fritsch-Carlson method, which guarantees monotonicity preservation.
 *
 * Behavior outside the defined time range:
 * - Returns zero for times before the first knot or after the last knot
 * - No extrapolation (safe default to prevent numerical blow-up)
 *
 * Template parameter T is the numeric type (double, AD, or AD2).
 */
template<typename T>
struct PchipForcing {
  std::vector<double> t;           // Knot times
  std::vector<double> a, b, c, d;  // Cubic coefficients per interval

  PchipForcing() = default;

  /**
   * Construct from raw time/value data.
   * Computes PCHIP coefficients using Fritsch-Carlson method.
   */
  PchipForcing(const std::vector<double>& times,
               const std::vector<double>& values) {
    initialize(times, values);
  }

  /**
   * Initialize from raw data (for deferred construction).
   *
   * @param times Vector of time points (will be sorted internally)
   * @param values Vector of forcing values corresponding to times
   *
   * @throws std::invalid_argument if:
   *   - fewer than 2 data points
   *   - times and values have different lengths
   *   - duplicate time points exist
   */
  void initialize(const std::vector<double>& times,
                  const std::vector<double>& values) {
    size_t n = times.size();
    if (n < 2) {
      throw std::invalid_argument("PchipForcing: need at least 2 data points");
    }
    if (values.size() != n) {
      throw std::invalid_argument("PchipForcing: times and values must have same length");
    }

    // Copy and sort by time (create sorted indices)
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&times](size_t a, size_t b) { return times[a] < times[b]; });

    t.resize(n);
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
      t[i] = times[idx[i]];
      y[i] = values[idx[i]];
    }

    // Check for duplicates (after sorting, duplicates are adjacent)
    for (size_t i = 0; i < n - 1; ++i) {
      if (t[i] == t[i + 1]) {
        throw std::invalid_argument("PchipForcing: duplicate time points not allowed");
      }
    }

    size_t n_intervals = n - 1;

    // Compute interval widths and slopes
    std::vector<double> h(n_intervals);
    std::vector<double> delta(n_intervals);
    for (size_t i = 0; i < n_intervals; ++i) {
      h[i] = t[i + 1] - t[i];
      // Note: h[i] > 0 guaranteed since we checked for duplicates above
      delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    // Compute slopes at each knot (Fritsch-Carlson method)
    std::vector<double> m(n, 0.0);

    // Interior points: weighted harmonic mean of adjacent slopes
    for (size_t i = 1; i < n - 1; ++i) {
      if (delta[i - 1] * delta[i] > 0) {
        // Same sign: use weighted harmonic mean
        double w1 = 2.0 * h[i] + h[i - 1];
        double w2 = h[i] + 2.0 * h[i - 1];
        m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
      } else {
        // Different signs or zero: set slope to zero (preserves monotonicity)
        m[i] = 0.0;
      }
    }

    // End points: one-sided differences
    m[0] = delta[0];
    m[n - 1] = delta[n_intervals - 1];

    // Ensure monotonicity at endpoints (Fritsch-Carlson conditions)
    if (n_intervals >= 1) {
      // Left endpoint
      if (delta[0] * m[0] <= 0) {
        m[0] = 0.0;
      } else if (std::abs(m[0]) > 3.0 * std::abs(delta[0])) {
        m[0] = 3.0 * delta[0];
      }

      // Right endpoint
      if (delta[n_intervals - 1] * m[n - 1] <= 0) {
        m[n - 1] = 0.0;
      } else if (std::abs(m[n - 1]) > 3.0 * std::abs(delta[n_intervals - 1])) {
        m[n - 1] = 3.0 * delta[n_intervals - 1];
      }
    }

    // Compute cubic coefficients for each interval
    // p(t) = a + b*(t - t_i) + c*(t - t_i)^2 + d*(t - t_i)^3
    a.resize(n_intervals);
    b.resize(n_intervals);
    c.resize(n_intervals);
    d.resize(n_intervals);

    for (size_t i = 0; i < n_intervals; ++i) {
      a[i] = y[i];
      b[i] = m[i];
      c[i] = (3.0 * delta[i] - 2.0 * m[i] - m[i + 1]) / h[i];
      d[i] = (m[i] + m[i + 1] - 2.0 * delta[i]) / (h[i] * h[i]);
    }
  }

  /**
   * Evaluate forcing at given time.
   * Returns zero outside the defined time range (no extrapolation).
   * Uses Horner's method for numerical stability within range.
   *
   * @param time Time at which to evaluate the forcing
   * @return Forcing value (zero if outside [t.front(), t.back()])
   */
  T operator()(const T& time) const {
    double td = extract_double(time);

    // Return zero outside defined range (safe default)
    if (td < t.front() || td > t.back()) {
      return T(0.0);
    }

    int i = find_interval(td);
    T dt = time - t[i];
    // Horner's method: a + dt*(b + dt*(c + dt*d))
    return a[i] + dt * (b[i] + dt * (c[i] + dt * d[i]));
  }

  /**
   * Time derivative of forcing (for Jacobian df/dt computation).
   * Returns zero outside the defined time range.
   * p(t) = a + b*dt + c*dt^2 + d*dt^3
   * p'(t) = b + 2*c*dt + 3*d*dt^2
   *
   * @param time Time at which to evaluate the derivative
   * @return Derivative value (zero if outside [t.front(), t.back()])
   */
  T derivative(const T& time) const {
    double td = extract_double(time);

    // Return zero outside defined range (safe default)
    if (td < t.front() || td > t.back()) {
      return T(0.0);
    }

    int i = find_interval(td);
    T dt = time - t[i];
    // Horner's method: b + dt*(2c + dt*3d)
    return b[i] + dt * (2.0 * c[i] + dt * 3.0 * d[i]);
  }

  /**
   * Get the minimum time in the forcing data.
   * @return First time point
   */
  double time_min() const { return t.front(); }

  /**
   * Get the maximum time in the forcing data.
   * @return Last time point
   */
  double time_max() const { return t.back(); }

  /**
   * Check if a time is within the defined range.
   * @param time Time to check
   * @return true if time is in [t.front(), t.back()]
   */
  bool in_range(double time) const {
    return time >= t.front() && time <= t.back();
  }

private:
  /**
   * Extract double value from numeric type (for interval search).
   * Uses SFINAE to recursively call .x() for any AD-like type.
   */
  static double extract_double(double x) { return x; }

  // For any AD-like type with .x() method (F<T>, F<F<T>>, etc.)
  template<typename U>
  static double extract_double(const U& x) {
    if constexpr (std::is_same_v<U, double>) {
      return x;
    } else {
      return extract_double(const_cast<U&>(x).x());
    }
  }

  /**
   * Find interval index i such that t[i] <= td < t[i+1].
   * Uses binary search, returns clamped index for boundary cases.
   *
   * @pre td is within [t.front(), t.back()] (caller must check)
   */
  int find_interval(double td) const {
    auto it = std::upper_bound(t.begin(), t.end(), td);
    int i = static_cast<int>(it - t.begin()) - 1;
    // Clamp to valid interval range (handles td == t.back() case)
    if (i < 0) i = 0;
    if (i >= static_cast<int>(t.size()) - 1) i = static_cast<int>(t.size()) - 2;
    return i;
  }
};

} // namespace cppode

#endif // CPPODE_PCHIP_FORCING_HPP
