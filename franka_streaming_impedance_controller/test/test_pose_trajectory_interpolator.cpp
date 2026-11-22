// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.
//
// UMI reference values for unchanged behavior, plus position, continuity and speed regressions.

#include <franka_streaming_impedance_controller/pose_trajectory_interpolator.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

using franka_streaming_impedance::Pose;
using franka_streaming_impedance::PoseTrajectoryInterpolator;
using franka_streaming_impedance::poseDistance;

namespace {

constexpr double kTol = 1e-9;
constexpr double kInf = std::numeric_limits<double>::infinity();

Pose makePose(double x, double y, double z, double yaw) {
  Pose p;
  p.position = Eigen::Vector3d(x, y, z);
  p.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return p;
}

void expectPose(const Pose& got, double x, double y, double z, double qz, double qw) {
  EXPECT_NEAR(got.position.x(), x, kTol);
  EXPECT_NEAR(got.position.y(), y, kTol);
  EXPECT_NEAR(got.position.z(), z, kTol);
  // Compare on the shorter arc: q and -q are the same rotation.
  const double sign = (got.orientation.w() < 0.0) ? -1.0 : 1.0;
  EXPECT_NEAR(sign * got.orientation.z(), qz, kTol);
  EXPECT_NEAR(sign * got.orientation.w(), qw, kTol);
  EXPECT_NEAR(got.orientation.x(), 0.0, kTol);
  EXPECT_NEAR(got.orientation.y(), 0.0, kTol);
}

// The fixture UMI's numbers were generated from: (0,0,0)@t=10 -> (1,2,3) yawed 90deg @t=11.
PoseTrajectoryInterpolator twoWaypoints() {
  return PoseTrajectoryInterpolator({10.0, 11.0},
                                    {makePose(0, 0, 0, 0), makePose(1, 2, 3, M_PI / 2)});
}

Eigen::Vector3d positionVelocity(const PoseTrajectoryInterpolator& interp, double t) {
  constexpr double dt = 1e-6;
  return (interp(t + dt).position - interp(t - dt).position) / (2.0 * dt);
}

}  // namespace

TEST(PoseTrajectoryInterpolator, ConstantTrajectoryHoldsItsSeed) {
  // The activation case: seeded at the arm's current pose, it must command that pose at every
  // instant until a chunk arrives. Anything else moves the arm the moment the controller starts.
  const PoseTrajectoryInterpolator interp(10.0, makePose(0.3, -0.1, 0.5, 0.25));

  for (const double t : {0.0, 10.0, 1e6}) {
    expectPose(interp(t), 0.3, -0.1, 0.5, std::sin(0.125), std::cos(0.125));
  }
}

TEST(PoseTrajectoryInterpolator, MatchesUmiLerpAndSlerp) {
  const auto interp = twoWaypoints();

  expectPose(interp(10.0), 0.0, 0.0, 0.0, 0.0, 1.0);
  expectPose(interp(10.25), 0.25, 0.5, 0.75, 0.195090322016, 0.980785280403);
  expectPose(interp(10.5), 0.5, 1.0, 1.5, 0.382683432365, 0.923879532511);
  expectPose(interp(11.0), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
}

TEST(PoseTrajectoryInterpolator, ClampsRatherThanExtrapolating) {
  // Running off either end must hold the endpoint. Extrapolating past the last waypoint is how a
  // stalled action stream turns into the arm accelerating away from the workspace.
  const auto interp = twoWaypoints();

  expectPose(interp(9.5), 0.0, 0.0, 0.0, 0.0, 1.0);
  expectPose(interp(11.5), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
}

TEST(PoseTrajectoryInterpolator, ScheduleWaypointSplicesWithoutDiscontinuity) {
  // The whole reason this class exists. Splicing at curr_time=10.5 must leave the trajectory
  // evaluating to exactly what it did there a moment ago; a jump is a jerk on real hardware.
  const auto interp = twoWaypoints();
  const Pose before = interp(10.5);

  const auto spliced = interp.scheduleWaypoint(makePose(5, 0, 0, 0), 12.0, 10.5, 11.0, kInf, kInf);

  ASSERT_EQ(spliced.size(), 3u);
  EXPECT_NEAR(spliced.firstTime(), 10.5, kTol);
  EXPECT_NEAR(spliced.lastTime(), 12.0, kTol);

  const Pose after = spliced(10.5);
  EXPECT_NEAR((after.position - before.position).norm(), 0.0, kTol);
  EXPECT_NEAR(after.orientation.angularDistance(before.orientation), 0.0, kTol);

  // Knots are not interpolated, so UMI's values still hold at t=11.0 (knot B) and t=12.0.
  expectPose(spliced(11.0), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
  expectPose(spliced(12.0), 5.0, 0.0, 0.0, 0.0, 1.0);

  // B retains its incoming velocity (1,2,3), while the new segment exits at (4,-2,-3).
  expectPose(spliced(11.5), 2.625, 1.5, 2.25, 0.382683432365, 0.923879532511);
}

TEST(PoseTrajectoryInterpolator, AppendPreservesTheAlreadyPlannedCurve) {
  PoseTrajectoryInterpolator interp(10.0, makePose(0, 0, 0, 0));
  interp = interp.scheduleWaypoint(makePose(1, 0, 0, 0), 11.0, 10.0, 10.0, kInf, kInf);
  const auto appended = interp.scheduleWaypoint(makePose(-1, 0, 0, 0), 12.0, 10.5, 11.0, kInf, kInf);
  for (double t = 10.5; t <= 11.0; t += 0.025) {
    EXPECT_NEAR((appended(t).position - interp(t).position).norm(), 0.0, kTol);
  }
}

TEST(PoseTrajectoryInterpolator, BatchConstructorDoesNotOvershootPastItsWaypoints) {
  // The failure mode an unclamped Catmull-Rom/Hermite spline is prone to: swinging past a knot's
  // own neighbors on the way through it. Each waypoint here backtracks in x, which is exactly the
  // shape (a local extremum at the interior knot) that must clamp to a zero tangent and hug the
  // path instead of bulging outward.
  const PoseTrajectoryInterpolator interp(
      {0.0, 1.0, 2.0, 3.0},
      {makePose(0, 0, 0, 0), makePose(1, 0, 0, 0), makePose(0.5, 0, 0, 0), makePose(1.5, 0, 0, 0)});

  for (double t = 0.0; t <= 3.0; t += 0.01) {
    const double x = interp(t).position.x();
    EXPECT_GE(x, -kTol) << "undershot below the path's minimum at t=" << t;
    EXPECT_LE(x, 1.5 + kTol) << "overshot past the path's maximum at t=" << t;
  }
}

TEST(PoseTrajectoryInterpolator, WaypointInThePastIsIgnored) {
  // Chunks cross a network and arrive late. A waypoint whose instant has passed must not become
  // the arm's new target, or every delayed chunk yanks it backwards.
  const auto interp = twoWaypoints();

  const auto same = interp.scheduleWaypoint(makePose(9, 9, 9, 0), 10.4, 10.5, 11.0, kInf, kInf);

  ASSERT_EQ(same.size(), 2u);
  EXPECT_NEAR(same.firstTime(), 10.0, kTol);
  EXPECT_NEAR(same.lastTime(), 11.0, kTol);
}

TEST(PoseTrajectoryInterpolator, SpeedLimitStretchesTheSegment) {
  // The 10 m displacement needs at least 10 s at the requested 1 m/s exit-speed limit.
  const auto interp = twoWaypoints();

  const auto limited =
      interp.scheduleWaypoint(makePose(11, 2, 3, M_PI / 2), 11.5, 10.5, 11.0, 1.0, kInf);

  ASSERT_EQ(limited.size(), 3u);
  EXPECT_NEAR(limited.lastTime(), 21.0, kTol);
  expectPose(limited(21.0), 11.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
}

TEST(PoseTrajectoryInterpolator, PoseDistanceSplitsTranslationFromRotation) {
  const auto [pos, rot] = poseDistance(makePose(0, 0, 0, 0), makePose(3, 4, 0, M_PI / 2));

  EXPECT_NEAR(pos, 5.0, kTol);
  EXPECT_NEAR(rot, M_PI / 2, kTol);
}

// The speed limit is ours, not UMI's — upstream splices with max_pos_speed = max_rot_speed = inf.
// That divergence breaks an invariant UMI relies on: its caller stores the REQUESTED waypoint time
// in `last_waypoint_time` (franka_interpolation_controller.py:351), which upstream is always the
// trajectory's tail because nothing ever stretches a segment. Once a limit can stretch one, the
// caller's record and the real tail are different numbers, and `scheduleWaypoint` uses the record
// for two things at once: detecting a rewind, and bounding the trim window.
//
// These two tests pin what that divergence actually costs, because it is not obvious from reading:
// the path keeps its shape, and the speed bound holds, but the reference's LAG is unbounded.

TEST(PoseTrajectoryInterpolator, SpeedLimitedChunkKeepsItsPathShape) {
  // A right-angle chunk: the first waypoint is 1.5 m out along +x, unreachable in the 0.5 s asked
  // for at 1 m/s, so the limiter engages and stays engaged; the rest of the chunk turns and runs
  // along +y. The corner is what discriminates. Feeding the trajectory's real tail back in as
  // `last_waypoint_time` instead of the requested time sends every following waypoint down the
  // rewind branch, which collapses the whole chunk to one straight segment from the origin to the
  // last pose — the arm would cut the corner. Here the first leg must still be a straight run
  // along +x with y pinned at zero.
  const double t0 = 0.0;
  PoseTrajectoryInterpolator interp(t0, makePose(0, 0, 0, 0));
  double last_waypoint = t0;

  for (int k = 0; k < 6; ++k) {
    const double target = t0 + 0.5 + 0.1 * k;
    interp = interp.scheduleWaypoint(makePose(1.5, 0.05 * k, 0, 0), target, t0, last_waypoint, 1.0,
                                     kInf);
    last_waypoint = target;
  }

  // The leading leg stays on +x and starts from rest instead of instantly reaching the cap.
  for (const double t : {0.25, 0.5}) {
    EXPECT_GT(interp(t).position.x(), 0.0);
    EXPECT_LE(interp(t).position.x(), t + kTol);
    EXPECT_NEAR(interp(t).position.y(), 0.0, 1e-6);
    EXPECT_LE(positionVelocity(interp, t).norm(), 1.0 + 1e-8);
  }
  EXPECT_GT(interp(0.5).position.x(), interp(0.25).position.x());

  // It still arrives at the chunk's final pose, and holds there.
  EXPECT_NEAR(interp(interp.lastTime()).position.x(), 1.5, kTol);
  EXPECT_NEAR(interp(interp.lastTime()).position.y(), 0.25, kTol);
}

TEST(PoseTrajectoryInterpolator, SpeedLimitBoundsReferenceSpeedButNotItsLag) {
  // The speed limit does not cap how far behind the policy the reference may fall:
  // a stream commanding 1.5 m/s against a 1.0 m/s limit accumulates the 0.5 m/s deficit forever, so
  // the trajectory's tail runs ever further into the future and the arm keeps crawling toward a
  // stale target long after the policy has stopped.
  //
  // Pinned deliberately rather than left to be discovered on the arm. If a cap on how far the tail
  // may lead `now` is ever added, this is the test that must change, and changing it should be a
  // decision rather than a surprise.
  constexpr double kMaxSpeed = 1.0;
  double curr = 0.0;
  double last_waypoint = 0.0;
  PoseTrajectoryInterpolator interp(curr, makePose(0, 0, 0, 0));

  for (int chunk = 0; chunk < 40; ++chunk) {
    curr = 0.3 * chunk;
    for (int k = 0; k < 16; ++k) {
      const double target = curr + 0.1 * (k + 1);
      interp = interp.scheduleWaypoint(makePose(0.15 * (chunk * 3 + k + 1), 0, 0, 0), target, curr,
                                       last_waypoint, kMaxSpeed, kInf);
      last_waypoint = target;
    }
  }

  // The reference never outruns the limit, anywhere on the trajectory.
  for (double t = curr; t + 0.1 <= interp.lastTime(); t += 0.1) {
    const double travelled = (interp(t + 0.1).position - interp(t).position).norm();
    EXPECT_LE(travelled, kMaxSpeed * 0.1 + 1e-9) << "reference exceeded max_pos_speed at t=" << t;
  }

  // Knots stay bounded — the trim window discards what `curr` has passed, so a stream that limits
  // forever does not grow the trajectory forever.
  EXPECT_LE(interp.size(), 20u);

  // But the backlog does grow without bound: ~0.5 s of deficit per second of streaming.
  EXPECT_GT(interp.lastTime() - curr, 5.0) << "expected the reference to fall behind; if this now "
                                              "fails, a lead cap was added and the comment above "
                                              "is stale";
}

TEST(PoseTrajectoryInterpolator, RepeatedSplicesDoNotGrowTheTrajectory) {
  // A 10 Hz chunk stream splices continuously for the length of an episode, so every splice runs in
  // a subscription callback that also frees the previous trajectory. The trim window is what keeps
  // that bounded: it discards whatever curr_time has passed, so the knot count settles instead of
  // tracking the episode's length.
  auto interp = twoWaypoints();

  for (int i = 0; i < 200; ++i) {
    const double now = 10.5 + 0.1 * i;
    interp = interp.scheduleWaypoint(makePose(0.01 * i, 0, 0, 0), now + 0.3, now, now + 0.2,
                                     kInf, kInf);
    ASSERT_LE(interp.size(), 8u) << "spliced trajectory is growing without bound";
  }
  EXPECT_GT(interp.size(), 1u) << "test is vacuous if the splices are not landing";
}

TEST(PoseTrajectoryInterpolator, ReversingWaypointsRespectTheEuclideanSpeedLimit) {
  constexpr double max_speed = 0.1;
  for (const Eigen::Vector3d& direction :
       {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 2, -2).normalized()}) {
    PoseTrajectoryInterpolator interp(0.0, makePose(0, 0, 0, 0));
    for (int i = 1; i <= 3; ++i) {
      Pose target;
      target.position = (i % 2) * 0.1 * direction;
      interp = interp.scheduleWaypoint(target, i, 0.0, i - 1, max_speed, kInf);
    }
    ASSERT_GT(interp.lastTime(), 0.0);
    EXPECT_NEAR((interp(interp.lastTime()).position - 0.1 * direction).norm(), 0.0, kTol);
    double peak_speed = 0.0;
    for (int i = 1; i < 2000; ++i) {
      peak_speed = std::max(peak_speed, positionVelocity(interp, interp.lastTime() * i / 2000).norm());
    }
    EXPECT_LE(peak_speed, max_speed + 1e-8);
  }
}

TEST(PoseTrajectoryInterpolator, TrimPreservesNonlinearPositionAndVelocity) {
  const PoseTrajectoryInterpolator interp(
      {0.0, 1.0, 2.0, 3.0},
      {makePose(0, 0, 0, 0), makePose(0.1, -0.03, 0, 0),
       makePose(0, 0.07, 0, 0), makePose(0.1, 0.02, 0, 0)});
  const auto trimmed = interp.trim(0.25, 2.75);
  double position_error = 0.0;
  double velocity_error = 0.0;
  for (int i = 0; i <= 24; ++i) {
    const double t = 0.3 + i * 0.1;
    position_error = std::max(position_error, (trimmed(t).position - interp(t).position).norm());
    velocity_error = std::max(velocity_error,
                             (positionVelocity(trimmed, t) - positionVelocity(interp, t)).norm());
  }
  EXPECT_LE(position_error, 1e-12);
  EXPECT_LE(velocity_error, 1e-8);
}

TEST(PoseTrajectoryInterpolator, AppendPreservesLiveVelocityInsideACurvedSegment) {
  const PoseTrajectoryInterpolator interp(
      {0.0, 1.0, 2.0, 3.0},
      {makePose(0, 0, 0, 0), makePose(0.1, 0, 0, 0),
       makePose(0, 0, 0, 0), makePose(0.1, 0, 0, 0)});
  constexpr double now = 0.5;
  constexpr double dt = 1e-6;
  const auto appended = interp.scheduleWaypoint(makePose(0.2, 0, 0, 0), 4.0, now, 3.0, 0.2, kInf);
  const Eigen::Vector3d before = (interp(now).position - interp(now - dt).position) / dt;
  const Eigen::Vector3d after = (appended(now + dt).position - appended(now).position) / dt;
  EXPECT_NEAR((appended(now).position - interp(now).position).norm(), 0.0, kTol);
  EXPECT_LE((after - before).norm(), 1e-6);
}

TEST(PoseTrajectoryInterpolator, RepeatedTrimPreservesMotionAndClampedHolds) {
  const PoseTrajectoryInterpolator interp(
      {0.0, 1.0, 2.0},
      {makePose(0, 0, 0, 0), makePose(0.1, 0.1, 0, 0), makePose(0.3, 0, 0, 0)});
  const auto extended = interp.trim(-1.0, 3.0);
  const auto trimmed = extended.trim(-0.75, 2.75);
  for (const double t : {-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5}) {
    EXPECT_NEAR((extended(t).position - interp(t).position).norm(), 0.0, kTol);
    EXPECT_NEAR((trimmed(t).position - interp(t).position).norm(), 0.0, kTol);
  }
  for (const double t : {-0.5, 0.5, 1.5, 2.5}) {
    EXPECT_LE((positionVelocity(trimmed, t) - positionVelocity(interp, t)).norm(), 1e-8);
  }
}

TEST(PoseTrajectoryInterpolator, LoweredSpeedLimitPreservesEntryAndLimitsExit) {
  const PoseTrajectoryInterpolator interp({0.0, 2.0},
                                         {makePose(0, 0, 0, 0), makePose(0.4, 0, 0, 0)});
  constexpr double now = 0.5;
  constexpr double dt = 1e-6;
  const auto slowed = interp.scheduleWaypoint(makePose(0.3, 0, 0, 0), 1.0, now, 2.0, 0.05, kInf);
  EXPECT_NEAR((slowed(now + dt).position - slowed(now).position).x() / dt, 0.2, 1e-6);
  for (int i = 1; i < 100; ++i) {
    const double t = now + (slowed.lastTime() - now) * i / 100;
    EXPECT_LE(positionVelocity(slowed, t).norm(), 0.2 + 1e-8);
  }
  const double end = slowed.lastTime();
  EXPECT_LE((slowed(end).position - slowed(end - dt).position).norm() / dt, 0.05 + 1e-6);
}

TEST(PoseTrajectoryInterpolator, SamePositionTargetPreservesEntryAndComesToRest) {
  const PoseTrajectoryInterpolator interp({0.0, 2.0},
                                         {makePose(0, 0, 0, 0), makePose(0.4, 0.2, 0, 0)});
  constexpr double now = 0.5;
  constexpr double dt = 1e-6;
  const auto stopped = interp.scheduleWaypoint(interp(now), 1.5, now, 2.0, 0.3, kInf);
  const Eigen::Vector3d entry = (stopped(now + dt).position - stopped(now).position) / dt;
  EXPECT_LE((entry - positionVelocity(interp, now)).norm(), 1e-6);
  const double end = stopped.lastTime();
  EXPECT_NEAR((stopped(end).position - interp(now).position).norm(), 0.0, kTol);
  EXPECT_LE((stopped(end).position - stopped(end - dt).position).norm() / dt, 1e-6);
}

TEST(PoseTrajectoryInterpolator, ReplansAtTheSameInstantPreserveNonzeroEntryVelocity) {
  const PoseTrajectoryInterpolator interp({0.0, 2.0},
                                         {makePose(0, 0, 0, 0), makePose(0.4, 0, 0, 0)});
  constexpr double now = 0.5;
  constexpr double dt = 1e-6;
  const auto first = interp.scheduleWaypoint(makePose(0.3, 0, 0, 0), 1.2, now, 2.0, 0.3, kInf);
  const auto second = first.scheduleWaypoint(makePose(0.2, 0, 0, 0), 1.0, now, 1.2, 0.3, kInf);
  EXPECT_DOUBLE_EQ(first.firstTime(), now);
  EXPECT_DOUBLE_EQ(second.firstTime(), now);
  EXPECT_NEAR((second(now).position - interp(now).position).norm(), 0.0, kTol);
  EXPECT_NEAR((second(now + dt).position - second(now).position).x() / dt, 0.2, 1e-6);
}

TEST(PoseTrajectoryInterpolator, ExtendingASinglePointTrimPreservesItsHold) {
  const PoseTrajectoryInterpolator interp({0.0, 2.0},
                                         {makePose(0, 0, 0, 0), makePose(0.2, 0, 0, 0)});
  const auto point = interp.trim(1.0, 1.0);
  const auto extended = point.trim(1.0, 2.0);
  for (const double t : {1.0, 1.25, 1.5, 1.75, 2.0}) {
    EXPECT_NEAR((extended(t).position - point(t).position).norm(), 0.0, kTol);
  }
}

TEST(PoseTrajectoryInterpolator, ReplanningAtAClampBoundaryPreservesIncomingVelocity) {
  const PoseTrajectoryInterpolator interp({0.0, 2.0},
                                         {makePose(0, 0, 0, 0), makePose(0.2, 0, 0, 0)});
  const auto held = interp.trim(-1.0, 3.0);
  const auto replanned = held.scheduleWaypoint(makePose(0.3, 0, 0, 0), 2.5, 2.0, 3.0, 0.5, kInf);
  constexpr double dt = 1e-6;
  const Eigen::Vector3d before = (held(2.0).position - held(2.0 - dt).position) / dt;
  const Eigen::Vector3d after = (replanned(2.0 + dt).position - replanned(2.0).position) / dt;
  EXPECT_LE((after - before).norm(), 1e-6);
}
