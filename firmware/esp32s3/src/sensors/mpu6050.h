#ifndef MARKUS_MPU6050_H
#define MARKUS_MPU6050_H

#include <Arduino.h>

struct MotionData {
    float ax, ay, az; // Acceleration in g
    float gx, gy, gz; // Angular velocity in deg/s
    bool rapid_head_motion;
};

class MarkusIMU {
private:
    bool m_available;
    uint8_t m_dev_addr;

public:
    MarkusIMU(uint8_t addr = 0x68);
    bool init();
    bool readMotion(MotionData& data);
    bool isAvailable() const { return m_available; }
};

#endif // MARKUS_MPU6050_H
