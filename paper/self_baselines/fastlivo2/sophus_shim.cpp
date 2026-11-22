// Minimal ABI-compatible stand-in for the classic Sophus (a621ff) shared
// library. FAST-LIVO only needs three out-of-line symbols from it.
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace Sophus {

class SO3 {
 public:
  SO3() : quaternion_(1, 0, 0, 0) {}
  explicit SO3(const Eigen::Quaterniond& q) : quaternion_(q) {}

  Eigen::Matrix3d matrix() const;
  Eigen::Quaterniond unit_quaternion() const { return quaternion_; }
  SO3 inverse() const { return SO3(quaternion_.conjugate()); }
  Eigen::Vector3d operator*(const Eigen::Vector3d& p) const {
    return quaternion_ * p;
  }

 private:
  Eigen::Quaterniond quaternion_;
};

class SE3 {
 public:
  SE3() : so3_(), trans_(Eigen::Vector3d::Zero()) {}
  SE3(const SO3& so3, const Eigen::Vector3d& t) : so3_(so3), trans_(t) {}

  SE3 inverse() const;
  Eigen::Vector3d operator*(const Eigen::Vector3d& p) const;
  Eigen::Matrix3d rotation_matrix() const { return so3_.matrix(); }
  Eigen::Vector3d translation() const { return trans_; }

 private:
  SO3 so3_;
  Eigen::Vector3d trans_;
};

Eigen::Matrix3d SO3::matrix() const {
  return quaternion_.toRotationMatrix();
}

SE3 SE3::inverse() const {
  SO3 inv = so3_.inverse();
  return SE3(inv, inv * (-trans_));
}

Eigen::Vector3d SE3::operator*(const Eigen::Vector3d& p) const {
  return so3_ * p + trans_;
}

}  // namespace Sophus
