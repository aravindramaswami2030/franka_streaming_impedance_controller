// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.
//
// C++/Eigen port of UMI's umi/common/pose_trajectory_interpolator.py, which is what turns a
// sparse action chunk into the continuous reference a 1 kHz servo needs.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <utility>
#include <vector>

namespace franka_streaming_impedance {

/// A pose on the trajectory: position plus orientation, both in the trajectory's own frame.
struct Pose {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

/// Translational distance (m) and angular distance (rad) between two poses.
std::pair<double, double> poseDistance(const Pose& a, const Pose& b);

/**
 * Cubic-Hermite position / slerp orientation interpolation over absolutely-timed waypoints.
 *
 * Trimming preserves the curve and its derivatives. Scheduled segments inherit entry velocity
 * and end at their own secant velocity, keeping position and velocity continuous at splices.
 * Reversing direction can therefore overshoot waypoint positions. Orientation is still slerp;
 * angular velocity can step at waypoints. Outside the horizon, the endpoint is held.
 *
 * Evaluation is allocation-free and safe to call from a realtime update(). Splicing is not — build
 * the new interpolator off the realtime thread and hand it over.
 *
 * Times are absolute seconds on whatever clock the caller uses; this class never reads a clock.
 */
class PoseTrajectoryInterpolator {
 public:
  /// Constant trajectory: evaluates to `pose` at every instant.
  PoseTrajectoryInterpolator(double time, const Pose& pose);

  /// Trajectory through `poses` at `times`, which must be non-empty, equal in length, and sorted.
  PoseTrajectoryInterpolator(std::vector<double> times, std::vector<Pose> poses);

  /// Interpolate at `t`, clamped to the trajectory's own endpoints (never extrapolates).
  Pose operator()(double t) const;

  double firstTime() const { return times_.front(); }
  double lastTime() const { return times_.back(); }
  std::size_t size() const { return times_.size(); }

  /// The sub-trajectory spanning [start_t, end_t], with both endpoints materialised as waypoints.
  PoseTrajectoryInterpolator trim(double start_t, double end_t) const;

  /**
   * Splice `pose` in as a waypoint at absolute instant `time`.
   *
   * Everything after `last_waypoint_time` is discarded, so a fresh action chunk supersedes the
   * tail of the previous one while the part already being executed is left alone. A waypoint at or
   * before `curr_time` is ignored — it refers to an instant that has passed.
   *
   * Speed limits stretch the new segment's duration to bound peak speed, not just average speed.
   * If the positional limit is reduced below entry speed, that inherited speed bounds the
   * transition and exit speed meets the new limit. Retained segments are not retimed.
   */
  PoseTrajectoryInterpolator scheduleWaypoint(const Pose& pose,
                                              double time,
                                              double curr_time,
                                              double last_waypoint_time,
                                              double max_pos_speed,
                                              double max_rot_speed) const;

 private:
  std::vector<double> times_;
  std::vector<Pose> poses_;
  std::vector<Eigen::Vector3d> velocities_;

  /// One-sided position derivative, including the clamped holds outside the horizon.
  Eigen::Vector3d velocity(double t, bool from_left = false) const;

  /// (Re)derive velocities_ from times_/poses_. Called once, by both constructors.
  void computeVelocities();
};

}  // namespace franka_streaming_impedance
