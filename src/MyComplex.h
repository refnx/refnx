#pragma once
#include <math.h> /* copysign */

namespace MyComplexNumber {
// Complex Class definition
class MyComplex {
public:
  double re;
  double im;

  MyComplex(double real = 0, double imaginary = 0) {
    re = real;
    im = imaginary;
  }
  // overload addition, subtraction, multiplication, division
  MyComplex &operator=(const MyComplex &comp) {
    re = comp.re;
    im = comp.im;

    return *this;
  }

  const MyComplex operator*(const double s) {

    return MyComplex(s * re, s * im);
  }

  friend const MyComplex operator*(MyComplex lhs, MyComplex rhs) {
    return MyComplex((lhs.re * rhs.re - lhs.im * rhs.im),
                     (lhs.re * rhs.im + lhs.im * rhs.re));
  }

  friend const MyComplex operator/(MyComplex lhs, MyComplex rhs) {
    double denom = rhs.re * rhs.re + rhs.im * rhs.im;

    return MyComplex((lhs.re * rhs.re + lhs.im * rhs.im) / denom,
                     (lhs.im * rhs.re - lhs.re * rhs.im) / denom);
  }

  friend MyComplex operator+(MyComplex lhs, MyComplex rhs) {
    return MyComplex(lhs.re + rhs.re, lhs.im + rhs.im);
  }

  friend const MyComplex operator-(const MyComplex lhs, const MyComplex rhs) {
    return MyComplex(lhs.re - rhs.re, lhs.im - rhs.im);
  }
};

// Complex functions
inline MyComplex compexp(MyComplex comp) {
  //	double exponent = exp(comp.re);
  return MyComplex(exp(comp.re) * cos(comp.im), exp(comp.re) * sin(comp.im));
}

inline MyComplex compcos(MyComplex comp) {
  return MyComplex(cos(comp.re) * cosh(comp.im),
                   0 - (sin(comp.re) * sinh(comp.im)));
}

inline MyComplex compsin(MyComplex comp) {
  return MyComplex(sin(comp.re) * cosh(comp.im), cos(comp.re) * sinh(comp.im));
}

inline double comparg(MyComplex comp) { return atan2(comp.im, comp.re); }

inline double compnorm(MyComplex comp) {
  return comp.im * comp.im + comp.re * comp.re;
}

inline MyComplex compsqrt(MyComplex comp) {
  // Use the half angle relation to calculate the square root. Prevents buffer
  // over/under flows. Don't change this code. This problem actually occurs
  double mag;
  double theta;
  double sqrtholder;

  if (comp.im != 0) {
    mag = sqrt(comp.re * comp.re + comp.im * comp.im);
    theta = atan2(comp.im, (comp.re + mag));
    sqrtholder = sqrt(mag);
    return MyComplex(sqrtholder * cos(theta), sqrtholder * sin(theta));
  }

  // floating point imaginary part is zero. Return the correct branch cut.
  if (comp.re < 0)
    return MyComplex(0, copysign(comp.im) * sqrt(fabs(comp.re)));

  return MyComplex(sqrt(comp.re), 0);
};

inline double compabs(MyComplex comp) {
  // While inefficient, this prevents buffer under/overflows - from Numerical
  // Recipes in C++
  double placeholder = 0;
  if (comp.re >= comp.im) {
    if (comp.re == 0)
      return fabs(comp.im);

    placeholder = comp.im / comp.re;
    return (fabs(comp.re) * sqrt(1.0 + placeholder * placeholder));
  } else {
    if (comp.im == 0)
      return fabs(comp.re);

    placeholder = comp.re / comp.im;
    return (fabs(comp.im) * sqrt(1.0 + placeholder * placeholder));
  }
};

inline MyComplex cln(MyComplex comp) {
  return MyComplex(log(compabs(comp)), comparg(comp));
}

inline void matmul(MyComplex a[2][2], MyComplex b[2][2], MyComplex c[2][2]) {

  //		c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0];
  //		c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1];
  //		c[1][0]	= a[1][0]*b[0][0] + a[1][1]*b[1][0];
  //		c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1];
  c[0][0].re = (a[0][0].re * b[0][0].re) + (a[0][1].re * b[1][0].re) -
               (a[0][0].im * b[0][0].im) - (a[0][1].im * b[1][0].im);
  c[0][1].re = (a[0][0].re * b[0][1].re) + (a[0][1].re * b[1][1].re) -
               (a[0][0].im * b[0][1].im) - (a[0][1].im * b[1][1].im);
  c[1][0].re = (a[1][0].re * b[0][0].re) + (a[1][1].re * b[1][0].re) -
               (a[1][0].im * b[0][0].im) - (a[1][1].im * b[1][0].im);
  c[1][1].re = (a[1][0].re * b[0][1].re) + (a[1][1].re * b[1][1].re) -
               (a[1][0].im * b[0][1].im) - (a[1][1].im * b[1][1].im);

  c[0][0].im = (a[0][0].re * b[0][0].im) + (a[0][1].re * b[1][0].im) +
               (a[0][0].im * b[0][0].re) + (a[0][1].im * b[1][0].re);
  c[0][1].im = (a[0][0].re * b[0][1].im) + (a[0][1].re * b[1][1].im) +
               (a[0][0].im * b[0][1].re) + (a[0][1].im * b[1][1].re);
  c[1][0].im = (a[1][0].re * b[0][0].im) + (a[1][1].re * b[1][0].im) +
               (a[1][0].im * b[0][0].re) + (a[1][1].im * b[1][0].re);
  c[1][1].im = (a[1][0].re * b[0][1].im) + (a[1][1].re * b[1][1].im) +
               (a[1][0].im * b[0][1].re) + (a[1][1].im * b[1][1].re);
}
}; // namespace MyComplexNumber
