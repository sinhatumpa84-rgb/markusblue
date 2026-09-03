#ifndef PTT_H
#define PTT_H

#include <Arduino.h>

class MarkusPTT {
private:
    int m_ptt_pin;
    int m_haptic_pin;
    bool m_last_state;
    bool m_debounced_state;
    uint32_t m_last_debounce_time;
    uint32_t m_debounce_delay_ms;

public:
    MarkusPTT(int ptt_pin = 1, int haptic_pin = 2, uint32_t debounce_delay_ms = 25);
    void init();
    bool update();
    bool isPressed() const { return m_debounced_state; }
    void pulseHaptic(uint32_t duration_ms = 80);
};

#endif // PTT_H
