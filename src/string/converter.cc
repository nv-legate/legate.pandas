/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <cmath>

#include "string/converter.h"

namespace legate {
namespace pandas {
namespace string {
namespace detail {

// The following is copied and adapted from cuDF

// significant digits is independent of scientific notation range
// digits more than this may require using long values instead of ints
static constexpr unsigned int significant_digits = 10;
// maximum power-of-10 that will fit in 32-bits
static constexpr unsigned int nine_digits = 1000000000;  // 1x10^9
// Range of numbers here is for normalizing the value.
// If the value is above or below the following limits, the output is converted to
// scientific notation in order to show (at most) the number of significant digits.
static constexpr double upper_limit = 1000000000;  // max is 1x10^9
static constexpr double lower_limit = 0.0001;      // printf uses scientific notation below this
// Tables for doing normalization: converting to exponent form
// IEEE double float has maximum exponent of 305 so these should cover everthing
const double upper10[9]  = {10, 100, 10000, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256};
const double lower10[9]  = {.1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256};
const double blower10[9] = {1.0, .1, .001, 1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255};

// utility for quickly converting known integer range to character array
void int2str(std::stringstream& ss, int value)
{
  if (value == 0) {
    ss << '0';
    return;
  }
  char buffer[significant_digits];  // should be big-enough for significant digits
  char* ptr = buffer;
  while (value > 0) {
    *ptr++ = (char)('0' + (value % 10));
    value /= 10;
  }
  while (ptr != buffer) ss << *--ptr;  // 54321 -> 12345
}

/**
 * @brief Dissect a float value into integer, decimal, and exponent components.
 *
 * @return The number of decimal places.
 */
int dissect_value(double value, unsigned int& integer, unsigned int& decimal, int& exp10)
{
  int decimal_places = significant_digits - 1;
  // normalize step puts value between lower-limit and upper-limit
  // by adjusting the exponent up or down
  exp10 = 0;
  if (value > upper_limit) {
    int fx = 256;
    for (int idx = 8; idx >= 0; --idx) {
      if (value >= upper10[idx]) {
        value *= lower10[idx];
        exp10 += fx;
      }
      fx = fx >> 1;
    }
  } else if ((value > 0.0) && (value < lower_limit)) {
    int fx = 256;
    for (int idx = 8; idx >= 0; --idx) {
      if (value < blower10[idx]) {
        value *= upper10[idx];
        exp10 -= fx;
      }
      fx = fx >> 1;
    }
  }
  //
  unsigned int max_digits = nine_digits;
  integer                 = (unsigned int)value;
  for (unsigned int i = integer; i >= 10; i /= 10) {
    --decimal_places;
    max_digits /= 10;
  }
  double remainder = (value - (double)integer) * (double)max_digits;
  decimal          = (unsigned int)remainder;
  remainder -= (double)decimal;
  decimal += (unsigned int)(2.0 * remainder);
  if (decimal >= max_digits) {
    decimal = 0;
    ++integer;
    if (exp10 && (integer >= 10)) {
      ++exp10;
      integer = 1;
    }
  }
  //
  while ((decimal % 10) == 0 && (decimal_places > 0)) {
    decimal /= 10;
    --decimal_places;
  }
  return decimal_places;
}

/**
 * @brief Main kernel method for converting float value to char output array.
 *
 * Output need not be more than (significant_digits + 7) bytes:
 * 7 = 1 sign, 1 decimal point, 1 exponent ('e'), 1 exponent-sign, 3 digits for exponent
 *
 * @param value Float value to convert.
 * @param output Memory to write output characters.
 * @return Number of bytes written.
 */
void float_to_string(std::stringstream& ss, double value)
{
  // check for valid value
  if (std::isnan(value)) ss << "NaN";
  bool bneg = false;
  if (std::signbit(value)) {  // handles -0.0 too
    value = -value;
    bneg  = true;
  }
  if (std::isinf(value)) {
    if (bneg)
      ss << "-Inf";
    else
      ss << "Inf";
    return;
  }

  // dissect value into components
  unsigned int integer = 0, decimal = 0;
  int exp10          = 0;
  int decimal_places = dissect_value(value, integer, decimal, exp10);
  //
  // now build the string from the
  // components: sign, integer, decimal, exp10, decimal_places
  //
  // sign
  if (bneg) ss << '-';
  // integer
  int2str(ss, integer);
  // decimal
  ss << '.';
  if (decimal_places) {
    char buffer[10];
    char* pb = buffer;
    while (decimal_places--) {
      *pb++ = (char)('0' + (decimal % 10));
      decimal /= 10;
    }
    while (pb != buffer)  // reverses the digits
      ss << *--pb;        // e.g. 54321 -> 12345
  } else
    ss << '0';  // always include at least .0
  // exponent
  if (exp10) {
    ss << 'e';
    if (exp10 < 0) {
      ss << '-';
      exp10 = -exp10;
    } else
      ss << '+';
    if (exp10 < 10) ss << '0';  // extra zero-pad
    int2str(ss, exp10);
  }
}

}  // namespace detail
}  // namespace string
}  // namespace pandas
}  // namespace legate
