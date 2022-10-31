#pragma once

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
};

template <typename T> inline Vector3<T> operator*(T s, const Vector3<T> &v) {
  return v * s;
}

typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
