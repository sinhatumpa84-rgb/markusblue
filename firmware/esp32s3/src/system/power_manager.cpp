#include "power_manager.h"
#include <algorithm>

MarkusPowerManager::MarkusPowerManager(int adc_pin, float ratio)
    : m_adc_pin(adc_pin), m_voltage_divider_ratio(ratio) {}

void MarkusPowerManager::init() {
    pinMode(m_adc_pin, INPUT);
    analogReadResolution(12); // 12-bit ADC (0 - 4095)
    analogSetAttenuation(ADC_11db); // Full 0 - 3.3V range
}

float MarkusPowerManager::getBatteryVoltage() {
    uint32_t raw = analogRead(m_adc_pin);
    float pin_voltage = ((float)raw / 4095.0f) * 3.3f;
    return pin_voltage * m_voltage_divider_ratio;
}

int MarkusPowerManager::getBatteryPercentage() {
    float v = getBatteryVoltage();
    // Li-Po curve mapping: 4.2V = 100%, 3.3V = 0%
    float pct = ((v - 3.3f) / (4.2f - 3.3f)) * 100.0f;
    return std::max(0, std::min(100, (int)pct));
}

bool MarkusPowerManager::isLowBattery() {
    return getBatteryVoltage() < 3.4f;
}
