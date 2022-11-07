#include "geometry.hpp"
#include <ostream>

static float Pi = 3.14159265358979323846;

inline float Radians(float deg) { return (Pi / 180) * deg; }

struct Matrix4x4 {
  // Matrix4x4 Public Methods
  Matrix4x4() {
    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
    m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
        m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
  }
  Matrix4x4(float mat[4][4]);
  Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11,
            float t12, float t13, float t20, float t21, float t22, float t23,
            float t30, float t31, float t32, float t33);
  bool operator==(const Matrix4x4 &m2) const {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if (m[i][j] != m2.m[i][j])
          return false;
    return true;
  }
  bool operator!=(const Matrix4x4 &m2) const {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if (m[i][j] != m2.m[i][j])
          return true;
    return false;
  }
  friend Matrix4x4 Transpose(const Matrix4x4 &);
  void Print(FILE *f) const {
    fprintf(f, "[ ");
    for (int i = 0; i < 4; ++i) {
      fprintf(f, "  [ ");
      for (int j = 0; j < 4; ++j) {
        fprintf(f, "%f", m[i][j]);
        if (j != 3)
          fprintf(f, ", ");
      }
      fprintf(f, " ]\n");
    }
    fprintf(f, " ] ");
  }
  static Matrix4x4 Mul(const Matrix4x4 &m1, const Matrix4x4 &m2) {
    Matrix4x4 r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
                    m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
    return r;
  }
  friend Matrix4x4 Inverse(const Matrix4x4 &);

  float m[4][4];
};

class Transform {
public:
  Transform(){};
  Transform(const Matrix4x4 &m) : m(m), mInv(Inverse(m)) {}
  Transform(const float mat[4][4]) {
    m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                  mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                  mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                  mat[3][3]);
    mInv = Inverse(m);
  }
  Transform(const Matrix4x4 &m, const Matrix4x4 &mInv) : m(m), mInv(mInv) {}

  friend Transform Inverse(const Transform &t) {
    return Transform(t.mInv, t.m);
  }

  friend Transform Transpose(const Transform &t) {
    return Transform(Transpose(t.m), Transpose(t.mInv));
  }
  Transform Translate(const Vector3f &delta) {
    Matrix4x4 m(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0,
                1);
    Matrix4x4 minv(1, 0, 0, -delta.x, 0, 1, 0, -delta.y, 0, 0, 1, -delta.z, 0,
                   0, 0, 1);
    return Transform(m, minv);
  }

  Transform Scale(float x, float y, float z) {
    Matrix4x4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
    Matrix4x4 minv(1 / x, 0, 0, 0, 0, 1 / y, 0, 0, 0, 0, 1 / z, 0, 0, 0, 0, 1);
    return Transform(m, minv);
  }

  bool HasScale() const {
    float la2 = (*this)(Vector3f(1, 0, 0)).LengthSquared();
    float lb2 = (*this)(Vector3f(0, 1, 0)).LengthSquared();
    float lc2 = (*this)(Vector3f(0, 0, 1)).LengthSquared();
#define NOT_ONE(x) ((x) < .999f || (x) > 1.001f)
    return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
  }

  Transform RotateX(float theta) {
    float sinTheta = std::sin(Radians(theta));
    float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0,
                0, 0, 0, 1);
    return Transform(m, Transpose(m));
  }

  Transform RotateZ(float theta) {
    float sinTheta = std::sin(Radians(theta));
    float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 1);
    return Transform(m, Transpose(m));
  }

  Transform RotateY(float theta) {
    float sinTheta = std::sin(Radians(theta));
    float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0,
                0, 0, 0, 1);
    return Transform(m, Transpose(m));
  }

  Transform Rotate(float theta, const Vector3f &axis) {
    Vector3f a = Normalize(axis);
    float sinTheta = std::sin(Radians(theta));
    float cosTheta = std::cos(Radians(theta));
    Matrix4x4 m;
    m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.m[0][3] = 0;

    m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.m[1][3] = 0;

    m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m.m[2][3] = 0;

    return Transform(m, Transpose(m));
  }

  Transform LookAt(const Point3f &pos, const Point3f &look,
                   const Vector3f &up) {
    Matrix4x4 cameraToWorld;
    cameraToWorld.m[0][3] = pos.x;
    cameraToWorld.m[1][3] = pos.y;
    cameraToWorld.m[2][3] = pos.z;
    cameraToWorld.m[3][3] = 1;

    Vector3f dir = Normalize(look - pos);
    Vector3f right = Normalize(Cross(Normalize(up), dir));
    Vector3f newUp = Cross(dir, right);
    cameraToWorld.m[0][0] = right.x;
    cameraToWorld.m[1][0] = right.y;
    cameraToWorld.m[2][0] = right.z;
    cameraToWorld.m[3][0] = 0.;
    cameraToWorld.m[0][1] = newUp.x;
    cameraToWorld.m[1][1] = newUp.y;
    cameraToWorld.m[2][1] = newUp.z;
    cameraToWorld.m[3][1] = 0.;
    cameraToWorld.m[0][2] = dir.x;
    cameraToWorld.m[1][2] = dir.y;
    cameraToWorld.m[2][2] = dir.z;
    cameraToWorld.m[3][2] = 0.;

    return Transform(Inverse(cameraToWorld), cameraToWorld);
  }

  template <typename T> Point3<T> operator()(const Point3<T> &p) const;
  template <typename T> Vector3<T> operator()(const Vector3<T> &v) const;
  template <typename T> Normal3<T> operator()(const Normal3<T> &v) const;
  template <typename T> Ray operator()(const Ray &v) const;

  Bounds3f operator()(const Bounds3f &b) const;
  Transform operator*(const Transform &t2) const;
  bool SwapHandedness() const;

private:
  Matrix4x4 m, mInv;
};
