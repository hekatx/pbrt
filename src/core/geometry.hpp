#pragma once
#include <algorithm>
#include <cmath>
#include <stdlib.h>

template <typename T> class Vector2 {
public:
  T x, y;
  Vector2(T x, T y) : x(x), y(y) {}
  T operator[](int i) const {
    if (i == 0)
      return x;
    if (i == 1)
      return y;
  }

  T &operator[](int i) {
    if (i == 0)
      return x;
    return y;
  };

  Vector2<T> operator+(const Vector2<T> &v) {
    return Vector2(x + v.x, y + v.y);
  }

  Vector2<T> operator-(const Vector2<T> &v) {
    return Vector2(x - v.x, y - v.y);
  }

  Vector2<T> operator*(T s) const { return Vector2<T>(s * x, s * y); }

  Vector2<T> &operator+=(const Vector2<T> &v) {
    x += v.x;
    y += v.y;
    return *this;
  }

  Vector2<T> &operator-=(const Vector2<T> &v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }

  Vector2<T> &operator*=(T s) {
    x *= s;
    y *= s;
    return *this;
  }

  Vector2<T> operator/(T f) const {
    float inv = (float)1 / f;
    return Vector2<T>(x * inv, y * inv);
  }

  Vector2<T> &operator/=(T f) {
    float inv = (float)1 / f;
    x *= inv;
    y *= inv;
    return *this;
  }

  Vector2<T> operator-() const { return Vector2<T>(-x, -y); }
};

template <typename T> class Vector3 {
public:
  T x, y, z;

  Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

  T operator[](int i) const {
    if (i == 0)
      return x;
    if (i == 1)
      return y;
    return z;
  }

  T &operator[](int i) {
    if (i == 0)
      return x;
    if (i == 1)
      return y;
    return z;
  }

  Vector3<T> operator+(const Vector3<T> &v) {
    return Vector3(x + v.x, y + v.y, z + v.z);
  }

  Vector3<T> operator-(const Vector3<T> &v) {
    return Vector3(x - v.x, y - v.y, z - v.z);
  }

  Vector3<T> operator*(T s) const { return Vector3<T>(s * x, s * y, s * z); }

  Vector3<T> &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  Vector3<T> &operator-=(const Vector3<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }

  Vector3<T> &operator*=(T s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }

  Vector3<T> operator/(T f) const {
    float inv = (float)1 / f;
    return Vector3<T>(x * inv, y * inv, z * inv);
  }

  Vector3<T> &operator/=(T f) {
    float inv = (float)1 / f;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
  }

  Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }

  float LengthSquared() const { return x * x + y * y + z * z; }
  float Length() const { return std::sqrt(LengthSquared()); }
};

template <typename T> inline Vector3<T> operator*(T s, const Vector3<T> &v) {
  return v * s;
}

template <typename T> inline T Dot(const Vector3<T> &v1, const Vector3<T> &v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T>
inline T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) {
  return std::abs(Dot(v1, v2));
}

template <typename T>
inline Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
  double v1x = v1.x, v1y = v1.y, v1z = v1.z;
  double v2x = v2.x, v2y = v2.y, v2z = v2.z;
  return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
                    (v1x * v2y) - (v1y * v2x));
}

template <typename T> inline Vector3<T> Normalize(const Vector3<T> &v) {
  return v / v.Length();
}

template <typename T> T MinComponent(const Vector3<T> &v) {
  return std::min(v.x, std::min(v.y, v.z));
}
template <typename T> T MaxComponent(const Vector3<T> &v) {
  return std::max(v.x, std::max(v.y, v.z));
}

template <typename T> int MaxDimension(const Vector3<T> &v) {
  return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

template <typename T>
Vector3<T> Min(const Vector3<T> &p1, const Vector3<T> &p2) {
  return Vector3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                    std::min(p1.z, p2.z));
}

template <typename T>
Vector3<T> Max(const Vector3<T> &p1, const Vector3<T> &p2) {
  return Vector3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                    std::max(p1.z, p2.z));
}

template <typename T>
inline void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2,
                             Vector3<T> *v3) {
  if (std::abs(v1.x) > std::abs(v1.y))
    *v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
  else
    *v2 = Vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
  *v3 = Cross(v1, *v2);
}

typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
