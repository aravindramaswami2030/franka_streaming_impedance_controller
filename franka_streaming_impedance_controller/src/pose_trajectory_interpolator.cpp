// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.

#include <franka_streaming_impedance_controller/pose_trajectory_interpolator.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

/// Shape-preserving tangents for a trajectory constructed from a complete set of waypoints.
Eigen::Vector3d monotoneTangent(const Eigen::Vector3d& secant_prev,
                                const Eigen::Vector3d& secant_next) {
  Eigen::Vector3d out;
  for (int i = 0; i < 3; ++i) {
    const double a = secant_prev(i);
    const double b = secant_next(i);
    if (a * b <= 0.0) {
      out(i) = 0.0;
      continue;
    }
    out(i) = std::copysign(std::min(std::abs(a), std::abs(b)), a);
  }
  return out;
}

}  // namespace

namespace franka_streaming_impedance {

std::pair<double, double> poseDistance(const Pose& a, const Pose& b) {
  return {(b.position - a.position).norm(), a.orientation.angularDistance(b.orientation)};
}

PoseTrajectoryInterpolator::PoseTrajectoryInterpolator(double time, const Pose& pose)
    : times_{time}, poses_{pose} {
  computeVelocities();
}

PoseTrajectoryInterpolator::PoseTrajectoryInterpolator(std::vector<double> times,
                                                       std::vector<Pose> poses)
    : times_(std::move(times)), poses_(std::move(poses)) {
  if (times_.empty() || times_.size() != poses_.size()) {
    throw std::invalid_argument("PoseTrajectoryInterpolator: need equal, non-empty times and poses");
  }
  if (!std::is_sorted(times_.begin(), times_.end())) {
    throw std::invalid_argument("PoseTrajectoryInterpolator: times must be non-decreasing");
  }
  computeVelocities();
}

void PoseTrajectoryInterpolator::computeVelocities() {
  const std::size_t n = times_.size();
  velocities_.assign(n, Eigen::Vector3d::Zero());
  if (n < 2) {
    return;
  }

  std::vector<Eigen::Vector3d> secants(n - 1, Eigen::Vector3d::Zero());
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const double span = times_[i + 1] - times_[i];
    if (span > 0.0) {
      secants[i] = (poses_[i + 1].position - poses_[i].position) / span;
    }
  }

  // Boundary knots have only one neighbor, so their tangent is that one chord's slope. For a
  // bare two-waypoint trajectory this is both endpoints, and the spline below reduces to lerp.
  velocities_.front() = secants.front();
  velocities_.back() = secants.back();
  for (std::size_t i = 1; i + 1 < n; ++i) {
    velocities_[i] = monotoneTangent(secants[i - 1], secants[i]);
  }
}

Pose PoseTrajectoryInterpolator::operator()(double t) const {
  t = std::clamp(t, times_.front(), times_.back());

  // First waypoint at or after t; the segment we want ends there.
  const auto upper = std::lower_bound(times_.begin(), times_.end(), t);
  if (upper == times_.begin()) {
    return poses_.front();
  }
  const std::size_t hi = static_cast<std::size_t>(upper - times_.begin());
  if (hi >= times_.size()) {
    return poses_.back();
  }
  const std::size_t lo = hi - 1;

  const double span = times_[hi] - times_[lo];
  // Coincident waypoints carry no direction; take the later one rather than dividing by zero.
  const double alpha = (span > 0.0) ? (t - times_[lo]) / span : 1.0;

  Pose out;
  if (span > 0.0) {
    // Cubic Hermite: matches both endpoint positions and velocities exactly, so the reference's
    // velocity is continuous across this knot instead of jumping here (see computeVelocities()).
    const double s = alpha;
    const double h00 = 2.0 * s * s * s - 3.0 * s * s + 1.0;
    const double h10 = s * s * s - 2.0 * s * s + s;
    const double h01 = -2.0 * s * s * s + 3.0 * s * s;
    const double h11 = s * s * s - s * s;
    out.position = h00 * poses_[lo].position + (h10 * span) * velocities_[lo] +
                   h01 * poses_[hi].position + (h11 * span) * velocities_[hi];
  } else {
    out.position = poses_[hi].position;
  }
  out.orientation = poses_[lo].orientation.slerp(alpha, poses_[hi].orientation);
  return out;
}

PoseTrajectoryInterpolator PoseTrajectoryInterpolator::trim(double start_t, double end_t) const {
  auto result = *this;
  result.times_.clear();
  result.poses_.clear();
  result.velocities_.clear();
  const auto push = [&](double t, const Pose& pose, const Eigen::Vector3d& velocity) {
    result.times_.push_back(t);
    result.poses_.push_back(pose);
    result.velocities_.push_back(velocity);
  };

  // A Hermite sub-segment needs the original derivatives as well as its endpoint positions.
  // A point splice uses incoming velocity; at the first knot, retain the existing outgoing one.
  push(start_t, (*this)(start_t), velocity(start_t, start_t == end_t && start_t > firstTime()));
  if (start_t < firstTime() && firstTime() < end_t) {
    push(firstTime(), poses_.front(), Eigen::Vector3d::Zero());
  }
  for (std::size_t i = 0; i < times_.size(); ++i) {
    if (times_[i] > start_t && times_[i] < end_t) {
      push(times_[i], poses_[i], velocities_[i]);
    }
  }
  // Duplicate boundary knots retain the jump to/from a clamped, zero-velocity hold.
  if (start_t < lastTime() && lastTime() < end_t) {
    push(lastTime(), poses_.back(), Eigen::Vector3d::Zero());
  }
  if (end_t > start_t) {
    push(end_t, (*this)(end_t), velocity(end_t, true));
  }

  return result;
}

Eigen::Vector3d PoseTrajectoryInterpolator::velocity(double t, bool from_left) const {
  const auto upper = from_left ? std::lower_bound(times_.begin(), times_.end(), t)
                               : std::upper_bound(times_.begin(), times_.end(), t);
  if (upper == times_.begin() || upper == times_.end()) {
    return Eigen::Vector3d::Zero();
  }
  const std::size_t hi = static_cast<std::size_t>(upper - times_.begin());
  const std::size_t lo = hi - 1;
  const double span = times_[hi] - times_[lo];
  const double s = (t - times_[lo]) / span;
  return 6.0 * s * (1.0 - s) / span * (poses_[hi].position - poses_[lo].position) +
         (3.0 * s * s - 4.0 * s + 1.0) * velocities_[lo] +
         (3.0 * s * s - 2.0 * s) * velocities_[hi];
}

PoseTrajectoryInterpolator PoseTrajectoryInterpolator::scheduleWaypoint(const Pose& pose,
                                                                       double time,
                                                                       double curr_time,
                                                                       double last_waypoint_time,
                                                                       double max_pos_speed,
                                                                       double max_rot_speed) const {
  // The waypoint refers to an instant that has already passed.
  if (time <= curr_time) {
    return *this;
  }

  // Window of the existing trajectory to keep. Ported from UMI verbatim: the sequence of min/max
  // operations is what establishes start_time <= end_time <= time and curr_time <= start_time,
  // which everything below relies on. Rearranging it is how you get a discontinuity at curr_time.
  double start_time = std::max(curr_time, times_.front());
  double end_time = (time <= last_waypoint_time) ? curr_time
                                                 : std::max(last_waypoint_time, curr_time);
  end_time = std::min(end_time, time);
  start_time = std::min(start_time, end_time);

  PoseTrajectoryInterpolator trimmed = trim(start_time, end_time);

  const Pose end_pose = trimmed(end_time);
  const auto [pos_dist, rot_dist] = poseDistance(pose, end_pose);
  const Eigen::Vector3d displacement = pose.position - end_pose.position;
  const Eigen::Vector3d entry_velocity = trimmed.velocities_.back();
  double duration = std::max({time - end_time, pos_dist / max_pos_speed,
                             rot_dist / max_rot_speed});
  if (pos_dist > 0.0 && std::isfinite(max_pos_speed)) {
    // With exit velocity d/T, the exact peak speed is max(|v0|, |4d/T - v0|/3).
    // A lowered limit cannot erase entry velocity; the secant bound above limits exit speed.
    const double limit = std::max(max_pos_speed, entry_velocity.norm());
    const double projection = displacement.dot(entry_velocity) / pos_dist;
    const double min_duration = 4.0 * pos_dist /
        (std::sqrt(projection * projection + 9.0 * limit * limit -
                   entry_velocity.squaredNorm()) + projection);
    duration = std::max(duration, min_duration);
  }

  trimmed.times_.push_back(end_time + duration);
  trimmed.poses_.push_back(pose);
  trimmed.velocities_.push_back(displacement / duration);
  return trimmed;
}

}  // namespace franka_streaming_impedance
