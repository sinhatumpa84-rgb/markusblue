#include "mpu6050.h"
#include <Wire.h>

#define MPU6050_PWR_MGMT_1   0x6B
#define MPU6050_ACCEL_XOUT_H 0x3B

MarkusIMU::MarkusIMU(uint8_t addr)
    : m_available(false), m_dev_addr(addr) {}

bool MarkusIMU::init() {
    Wire.beginTransmission(m_dev_addr);
    Wire.write(MPU6050_PWR_MGMT_1);
    Wire.write(0x00); // Wake up MPU6050
    uint8_t error = Wire.endTransmission();

    m_available = (error == 0);
    return m_available;
}

bool MarkusIMU::readMotion(MotionData& data) {
    if (!m_available) {
        data.ax = data.ay = data.az = 0.0f;
        data.gx = data.gy = data.gz = 0.0f;
        data.rapid_head_motion = false;
        return false;
    }

    Wire.beginTransmission(m_dev_addr);
    Wire.write(MPU6050_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    if (Wire.requestFrom((int)m_dev_addr, 14) != 14) {
        return false;
    }

    int16_t raw_ax = (Wire.read() << 8) | Wire.read();
    int16_t raw_ay = (Wire.read() << 8) | Wire.read();
    int16_t raw_az = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read(); // Skip temperature
    int16_t raw_gx = (Wire.read() << 8) | Wire.read();
    int16_t raw_gy = (Wire.read() << 8) | Wire.read();
    int16_t raw_gz = (Wire.read() << 8) | Wire.read();

    // Scale to standard units (+/- 2g accel, +/- 250 deg/s gyro)
    data.ax = (float)raw_ax / 16384.0f;
    data.ay = (float)raw_ay / 16384.0f;
    data.az = (float)raw_az / 16384.0f;

    data.gx = (float)raw_gx / 131.0f;
    data.gy = (float)raw_gy / 131.0f;
    data.gz = (float)raw_gz / 131.0f;

    // Detect rapid head turns (> 120 deg/sec)
    float gyro_mag = sqrtf(data.gx * data.gx + data.gy * data.gy + data.gz * data.gz);
    data.rapid_head_motion = (gyro_mag > 120.0f);

    return true;
}
