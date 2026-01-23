//
// Created by team204 on 3/3/25.
//

#include "swerve/swerve_module.h"

using namespace Constants::Swerve;

// for some reason, these are missing from wpilib cpp
double toRotations(frc::Rotation2d rot) {
    return rot.Degrees().value() / 360.0;
}
frc::Rotation2d fromRotations(double rot) {
    return frc::Rotation2d{units::degree_t{rot * 360.0}};
}

SwerveModule::SwerveModule(int moduleNumber, int driveMotorID, int angleMotorID, frc::Rotation2d angleOffset, int canCoderID) :
moduleNumber{moduleNumber},
angleOffset{angleOffset},
angleEncoder{canCoderID, "can1"},
angleMotor{angleMotorID, rev::spark::SparkLowLevel::MotorType::kBrushless},
integratedAngleEncoder{angleMotor.GetEncoder()},
angleController{angleMotor.GetClosedLoopController()},
driveMotor{driveMotorID, rev::spark::SparkLowLevel::MotorType::kBrushless},
driveEncoder{driveMotor.GetEncoder()},
driveController{driveMotor.GetClosedLoopController()},
feedforward{DRIVE_FF_S, DRIVE_FF_V, DRIVE_FF_A},
lastAngle{}
{
    ConfigAngleEncoder();
    ConfigAngleMotor();
    ConfigDriveMotor();

    lastAngle = GetState().angle;
}

void SwerveModule::ConfigAngleEncoder() {
    angleEncoderConfig.MagnetSensor.AbsoluteSensorDiscontinuityPoint = units::turn_t{1};
    angleEncoderConfig.MagnetSensor.SensorDirection = ctre::phoenix6::signals::SensorDirectionValue::CounterClockwise_Positive;
    angleEncoder.GetConfigurator().Apply(angleEncoderConfig);
}

void SwerveModule::ConfigAngleMotor() {
    angleConfig
        .SetIdleMode(rev::spark::SparkBaseConfig::IdleMode::kBrake)
        .SmartCurrentLimit(20)
        .Inverted(true);
    angleConfig.encoder
            .PositionConversionFactor(ANGLE_POSITION_CONVERSION_FACTOR) // radians
            .VelocityConversionFactor(1); // radians per second
    angleConfig.closedLoop
            .SetFeedbackSensor(rev::spark::ClosedLoopConfig::FeedbackSensor::kPrimaryEncoder)
            .Pid(1, 0, 0)
            .OutputRange(-1,1)
            .PositionWrappingEnabled(true)
            .PositionWrappingInputRange(0,1)
            .MinOutput(-1)
            .MaxOutput(1);
    angleConfig.closedLoop.Apply(angleConfig.closedLoop);
    angleConfig.Apply(angleConfig);

    angleConfig.VoltageCompensation(VOLTAGE_COMPENSATION);
    angleMotor.Configure(angleConfig, rev::spark::SparkBase::ResetMode::kResetSafeParameters, rev::spark::SparkBase::PersistMode::kPersistParameters);

    std::this_thread::sleep_for(std::literals::chrono_literals::operator ""s(0.2));
    ResetToAbsolute();
}

void SwerveModule::ConfigDriveMotor() {
    driveConfig.SmartCurrentLimit(DRIVE_CURRENT_LIMIT) ;
    driveConfig
        .Inverted(true)
        .SetIdleMode(rev::spark::SparkBaseConfig::IdleMode::kBrake);
    driveConfig.encoder
        .PositionConversionFactor(DRIVE_POSITION_CONVERSION_FACTOR)
        .VelocityConversionFactor(DRIVE_VELOCITY_CONVERSION_FACTOR);
    driveConfig.closedLoop
        .SetFeedbackSensor(rev::spark::ClosedLoopConfig::FeedbackSensor::kPrimaryEncoder)
        .Pidf(DRIVE_PID_P, DRIVE_PID_I, DRIVE_PID_D, DRIVE_PID_FF);

    driveConfig.VoltageCompensation(VOLTAGE_COMPENSATION);
    driveMotor.Configure(driveConfig, rev::spark::SparkBase::ResetMode::kResetSafeParameters, rev::spark::SparkBase::PersistMode::kPersistParameters);

    driveEncoder.SetPosition(0.0);
}


void SwerveModule::SetDesiredState(frc::SwerveModuleState desiredState, bool isOpenLoop) {
    desiredState.Optimize(GetState().angle); // desiredState = frc::SwerveModuleState::Optimize(desiredState, frc::Rotation2d(units::degree_t{integratedAngleEncoder.GetPosition()}));

    frc::Rotation2d angle = (units::math::abs(desiredState.speed) <= (MAX_SPEED * 0.01)) ? lastAngle : desiredState.angle;
    angleController.SetReference(toRotations(angle), rev::spark::SparkLowLevel::ControlType::kPosition);
    //SetAngle(desiredState);

    if (isOpenLoop) {
        driveMotor.Set(desiredState.speed / MAX_SPEED);
    } else {
        driveController.SetReference(
                desiredState.speed.value(),
                rev::spark::SparkLowLevel::ControlType::kVelocity,
                rev::spark::kSlot0,
                feedforward.Calculate(desiredState.speed).value());
    }
    //SetSpeed(desiredState, isOpenLoop);
}

void SwerveModule::ResetToAbsolute() {
    double absolutePosition = angleEncoder.GetAbsolutePosition().GetValueAsDouble() - toRotations(angleOffset);
    if (absolutePosition < 0) {
        absolutePosition += 1;
    }
    std::this_thread::sleep_for(std::literals::chrono_literals::operator ""s(0.3));
    integratedAngleEncoder.SetPosition(std::abs(absolutePosition));
    std::this_thread::sleep_for(std::literals::chrono_literals::operator ""s(0.3));
}

frc::Rotation2d SwerveModule::GetAngle() {
    return fromRotations(integratedAngleEncoder.GetPosition());
}

frc::SwerveModuleState SwerveModule::GetState() {
    return frc::SwerveModuleState{units::meters_per_second_t{driveEncoder.GetVelocity()}, GetAngle()};
}

frc::SwerveModulePosition SwerveModule::GetPosition() {
    return frc::SwerveModulePosition{units::length::meter_t{driveEncoder.GetPosition()}, GetAngle()};
}

void SwerveModule::ZeroDriveEncoder() {
    driveEncoder.SetPosition(0.0);
}
