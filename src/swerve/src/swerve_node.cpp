//
// Created by team204 on 4/9/25.
//

#include "swerve/swerve_node.h"

using namespace Constants::Swerve;

SwerveNode::SwerveNode() :
Node{"swerve"},
gyro{9, "can1"},
modules{
    SwerveModule{0, 11, 21, frc::Rotation2d{units::degree_t{0.75 * 360.0}}, 31},
    SwerveModule{1, 12, 22, frc::Rotation2d{units::degree_t{0.888 * 360.0}}, 32},
    SwerveModule{2, 10, 20, frc::Rotation2d{units::degree_t{0.887 * 360.0}}, 30},
    SwerveModule{3, 13, 23, frc::Rotation2d{units::degree_t{0.08 * 360.0}}, 33}
}
{
    gyro.GetConfigurator().Apply(ctre::phoenix6::configs::Pigeon2Configuration{});
    ZeroGyro();

    auto driveCbWrapper = [this](const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Request> request, const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Response> response) {
        DriveCb(request, response);
    };
    driveService = this->create_service<vikings_msgs::srv::DriveSwerve>("swerve/donotusethis", driveCbWrapper);

    auto zeroGyroCb = [this](const std::shared_ptr<vikings_msgs::srv::Blank::Request> request, const std::shared_ptr<vikings_msgs::srv::Blank::Response> response) {
        (void) request;
        (void) response;
        ZeroGyro();
    };
    zeroGyroService = this->create_service<vikings_msgs::srv::Blank>("swerve/zeroGyro", zeroGyroCb);

    auto zeroDriveEncodersCb = [this](const std::shared_ptr<vikings_msgs::srv::Blank::Request> request, const std::shared_ptr<vikings_msgs::srv::Blank::Response> response) {
        (void) request;
        (void) response;
        ZeroDriveEncoders();
    };
    zeroDriveEncodersService = this->create_service<vikings_msgs::srv::Blank>("swerve/zeroDriveEncoders", zeroDriveEncodersCb);

    this->actionServer = rclcpp_action::create_server<vikings_msgs::action::DriveSwerve>(
            this,
            "swerve/drive",
            [](auto uuid, auto goal) { (void) uuid; (void) goal; return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE; },
            [](auto goal_handle) { (void) goal_handle; return rclcpp_action::CancelResponse::ACCEPT; },
            [this](auto goal_handle) {
                (void) goal_handle;
                std::thread t([this, goal_handle]() {
                    this->Drive(goal_handle);
                });
                t.detach();
            }
            );
}

void SwerveNode::DriveCb(const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Request> request, const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Response> response) {
    (void) response;

    std::cout << "DRIVE: x: " << request->x << " y: " << request->y << " rot: " << request->rot << " field_relative: " << request->field_relative << " open_loop: " << request->open_loop << "\n";

    auto newStates = SWERVE_KINEMATICS.ToSwerveModuleStates(request->field_relative ?
        frc::ChassisSpeeds::FromFieldRelativeSpeeds(units::meters_per_second_t{request->x}, units::meters_per_second_t{request->y}, units::radians_per_second_t{request->rot}, GetYaw()) :
        frc::ChassisSpeeds{units::meters_per_second_t{request->x}, units::meters_per_second_t{request->y}, units::radians_per_second_t{request->rot}}
    );
    frc::SwerveDriveKinematics<4>::DesaturateWheelSpeeds(&newStates, MAX_SPEED);

    for (SwerveModule& mod : modules) {
        mod.SetDesiredState(newStates[mod.moduleNumber], request->open_loop);
    }
}

frc::Rotation2d SwerveNode::GetYaw() {
    return frc::Rotation2d{units::degree_t{360 - gyro.GetAngle()}};
}
void SwerveNode::ZeroGyro() {
    gyro.SetYaw(units::degree_t{0.0});
}
void SwerveNode::ZeroDriveEncoders() {
    for (SwerveModule& mod : modules) {
        mod.ZeroDriveEncoder();
    }
}


void SwerveNode::Drive(const GoalHandle h) {

}