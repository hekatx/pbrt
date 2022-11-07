#include "geometry.hpp"

struct Interaction {
  Point3f p;
  Normal3f n;
  Vector3f pError;
  Vector3f wo;
  float time;

  Interaction(const Point3f &p, const Normal3f &n, const Vector3f &pError,
              const Vector3f &wo, float time);

  Interaction();

  bool IsSurfaceInteraction() const;
};

class Shape {
public:
  static bool reverseOrientation;
  static bool transformSwapsHandedness;
};

class SurfaceInteraction : public Interaction {
public:
  Point2f uv;
  Vector3f dpdu, dpdv;
  Normal3f dndu, dndv;
  const Shape *shape = nullptr;

  struct {
    Normal3f n;
    Vector3f dpdu, dpdv;
    Normal3f dndu, dndv;
  } shading;

  SurfaceInteraction();

  SurfaceInteraction(const Point3f &p, const Vector3f &pError,
                     const Point2f &uv, const Vector3f &wo,
                     const Vector3f &dpdu, const Vector3f &dpdv,
                     const Normal3f &dndu, const Normal3f &dndv, float time,
                     const Shape *shape);

  void SetShadingGeometry(const Vector3f &dpdus, const Vector3f &dpdvs,
                          const Normal3f &dndus, const Normal3f &dndvs,
                          bool orientationIsAuthoritative);
};
