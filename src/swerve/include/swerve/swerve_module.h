//
// Created by team204 on 3/3/25.
//

#ifndef SWERVE_MODULE_H
#define SWERVE_MODULE_H

#include <frc/geometry/Rotation2d.h>
#include <rev/SparkMax.h>
#include <rev/config/SparkMaxConfig.h>
#include <ctre/phoenix6/CANcoder.hpp>
#include <frc/controller/SimpleMotorFeedforward.h>
#include <units/velocity.h>
#include <units/acceleration.h>
#include <frc/kinematics/SwerveDriveKinematics.h>
#include "swerve/swerve_constants.h"
#include <thread>
#include <chrono>

class SwerveModule {
public:
    SwerveModule(int moduleNumber, int driveMotorID, int angleMotorID, frc::Rotation2d angleOffset, int canCoderID);
    void SetDesiredState(frc::SwerveModuleState desiredState, bool isOpenLoop);
    void ResetToAbsolute();
    frc::Rotation2d GetAngle();
    frc::SwerveModuleState GetState();
    frc::SwerveModulePosition GetPosition();
    void ZeroDriveEncoder();
    const int moduleNumber;

    bool IsAtSetpoint();

private:
    frc::Rotation2d lastAngle;
    frc::Rotation2d angleOffset;

    rev::spark::SparkMax driveMotor;
    rev::spark::SparkMax angleMotor;
    rev::spark::SparkMaxConfig driveConfig;
    rev::spark::SparkMaxConfig angleConfig;
    rev::spark::SparkRelativeEncoder driveEncoder;
    rev::spark::SparkRelativeEncoder integratedAngleEncoder;
    ctre::phoenix6::hardware::CANcoder angleEncoder;
    ctre::phoenix6::configs::CANcoderConfiguration angleEncoderConfig;

    rev::spark::SparkClosedLoopController driveController;
    rev::spark::SparkClosedLoopController angleController;

    frc::SimpleMotorFeedforward<units::meters> feedforward;

    void ConfigAngleMotor();
    void ConfigAngleEncoder();
    void ConfigDriveMotor();
};

#endif //SWERVE_MODULE_H
