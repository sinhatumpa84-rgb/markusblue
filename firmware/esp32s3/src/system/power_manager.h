#ifndef POWER_MANAGER_H
#define POWER_MANAGER_H

#include <Arduino.h>

class MarkusPowerManager {
private:
    int m_adc_pin;
    float m_voltage_divider_ratio;

public:
    MarkusPowerManager(int adc_pin = 7, float ratio = 2.0f);
    void init();
    float getBatteryVoltage();
    int getBatteryPercentage();
    bool isLowBattery();
};

#endif // POWER_MANAGER_H
