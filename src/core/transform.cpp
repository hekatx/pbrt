#include "transform.hpp"
#include <cmath>
#include <cstring>

void Error(const std::string &) {}

// Matrix4x4 Method Definitions
bool SolveLinearSystem2x2(const float A[2][2], const float B[2], float *x0,
                          float *x1) {
  float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if (std::abs(det) < 1e-10f)
    return false;
  *x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
  *x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
  if (std::isnan(*x0) || std::isnan(*x1))
    return false;
  return true;
}

Matrix4x4::Matrix4x4(float mat[4][4]) { memcpy(m, mat, 16 * sizeof(float)); }

Matrix4x4::Matrix4x4(float t00, float t01, float t02, float t03, float t10,
                     float t11, float t12, float t13, float t20, float t21,
                     float t22, float t23, float t30, float t31, float t32,
                     float t33) {
  m[0][0] = t00;
  m[0][1] = t01;
  m[0][2] = t02;
  m[0][3] = t03;
  m[1][0] = t10;
  m[1][1] = t11;
  m[1][2] = t12;
  m[1][3] = t13;
  m[2][0] = t20;
  m[2][1] = t21;
  m[2][2] = t22;
  m[2][3] = t23;
  m[3][0] = t30;
  m[3][1] = t31;
  m[3][2] = t32;
  m[3][3] = t33;
}

Matrix4x4 Transpose(const Matrix4x4 &m) {
  return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
                   m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
                   m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
                   m.m[3][3]);
}

Matrix4x4 Inverse(const Matrix4x4 &m) {
  int indxc[4], indxr[4];
  int ipiv[4] = {0, 0, 0, 0};
  float minv[4][4];
  memcpy(minv, m.m, 4 * 4 * sizeof(float));
  for (int i = 0; i < 4; i++) {
    int irow = 0, icol = 0;
    float big = 0.f;
    // Choose pivot
    for (int j = 0; j < 4; j++) {
      if (ipiv[j] != 1) {
        for (int k = 0; k < 4; k++) {
          if (ipiv[k] == 0) {
            if (std::abs(minv[j][k]) >= big) {
              big = float(std::abs(minv[j][k]));
              irow = j;
              icol = k;
            }
          } else if (ipiv[k] > 1)
            Error("Singular matrix in MatrixInvert");
        }
      }
    }
    ++ipiv[icol];
    // Swap rows _irow_ and _icol_ for pivot
    if (irow != icol) {
      for (int k = 0; k < 4; ++k)
        std::swap(minv[irow][k], minv[icol][k]);
    }
    indxr[i] = irow;
    indxc[i] = icol;
    if (minv[icol][icol] == 0.f)
      Error("Singular matrix in MatrixInvert");

    // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
    float pivinv = 1. / minv[icol][icol];
    minv[icol][icol] = 1.;
    for (int j = 0; j < 4; j++)
      minv[icol][j] *= pivinv;

    // Subtract this row from others to zero out their columns
    for (int j = 0; j < 4; j++) {
      if (j != icol) {
        float save = minv[j][icol];
        minv[j][icol] = 0;
        for (int k = 0; k < 4; k++)
          minv[j][k] -= minv[icol][k] * save;
      }
    }
  }
  // Swap columns to reflect permutation
  for (int j = 3; j >= 0; j--) {
    if (indxr[j] != indxc[j]) {
      for (int k = 0; k < 4; k++)
        std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
    }
  }
  return Matrix4x4(minv);
}

Bounds3f Transform::operator()(const Bounds3f &b) const {
  const Transform &M = *this;
  Bounds3f ret(M(Point3f(b.pMin.x, b.pMin.y, b.pMin.z)));
  ret = Union(ret, M(Point3f(b.pMax.x, b.pMin.y, b.pMin.z)));
  ret = Union(ret, M(Point3f(b.pMin.x, b.pMax.y, b.pMin.z)));
  ret = Union(ret, M(Point3f(b.pMin.x, b.pMin.y, b.pMax.z)));
  ret = Union(ret, M(Point3f(b.pMin.x, b.pMax.y, b.pMax.z)));
  ret = Union(ret, M(Point3f(b.pMax.x, b.pMax.y, b.pMin.z)));
  ret = Union(ret, M(Point3f(b.pMax.x, b.pMin.y, b.pMax.z)));
  ret = Union(ret, M(Point3f(b.pMax.x, b.pMax.y, b.pMax.z)));
  return ret;
}

template <typename T>
inline Point3<T> Transform::operator()(const Point3<T> &p) const {
  T x = p.x, y = p.y, z = p.z;
  T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
  T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
  T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
  T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
  if (wp == 1)
    return Point3<T>(xp, yp, zp);
  else
    return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::operator()(const Vector3<T> &v) const {
  T x = v.x, y = v.y, z = v.z;
  return Vector3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                    m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                    m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

template <typename T>
inline Normal3<T> Transform::operator()(const Normal3<T> &n) const {
  T x = n.x, y = n.y, z = n.z;
  return Normal3<T>(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
                    mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
                    mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
}

Transform Transform::operator*(const Transform &t2) const {
  return Transform(Matrix4x4::Mul(m, t2.m), Matrix4x4::Mul(t2.mInv, mInv));
}

bool Transform::SwapsHandedness() const {
  float det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
              m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
              m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
  return det < 0;
}
