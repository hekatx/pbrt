#include "interaction.hpp"
#include "geometry.hpp"

Interaction::Interaction(const Point3f &p, const Normal3f &n,
                         const Vector3f &pError, const Vector3f &wo, float time)
    : p(p), n(n), pError(pError), wo(wo), time(time){};

Interaction::Interaction() : time(0){};

bool Interaction::IsSurfaceInteraction() const { return n != Normal3f(); }

SurfaceInteraction::SurfaceInteraction(
    const Point3f &p, const Vector3f &pError, const Point2f &uv,
    const Vector3f &wo, const Vector3f &dpdu, const Vector3f &dpdv,
    const Normal3f &dndu, const Normal3f &dndv, float time, const Shape *shape)
    : Interaction(p, Normal3f(Normalize(Cross(dpdu, dpdv))), pError, wo, time),
      uv(uv), dpdu(dpdu), dpdv(dpdv), dndu(dndu), dndv(dndv), shape(shape) {
  shading.n = n;
  shading.dndu = dndu;
  shading.dndv = dndv;
  shading.dpdu = dpdu;
  shading.dpdv = dpdv;

  if (shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness)) {
    n *= -1;
    shading.n *= -1;
  }
}

void SurfaceInteraction::SetShadingGeometry(const Vector3f &dpdus,
                                            const Vector3f &dpdvs,
                                            const Normal3f &dndus,
                                            const Normal3f &dndvs,
                                            bool orientationIsAuthoritative) {
  shading.n = Normalize((Normal3f)Cross(dpdus, dpdvs));
  if (shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness))
    shading.n = -shading.n;
  if (orientationIsAuthoritative)
    n = Faceforward(n, shading.n);
  else
    shading.n = Faceforward(shading.n, n);

  shading.dpdu = dpdus;
  shading.dpdv = dpdvs;
  shading.dndu = dndus;
  shading.dndv = dndvs;
};
