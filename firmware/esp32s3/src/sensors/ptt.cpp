#include "ptt.h"

MarkusPTT::MarkusPTT(int ptt_pin, int haptic_pin, uint32_t debounce_delay_ms)
    : m_ptt_pin(ptt_pin),
      m_haptic_pin(haptic_pin),
      m_last_state(HIGH),
      m_debounced_state(false),
      m_last_debounce_time(0),
      m_debounce_delay_ms(debounce_delay_ms) {}

void MarkusPTT::init() {
    pinMode(m_ptt_pin, INPUT_PULLUP);
    pinMode(m_haptic_pin, OUTPUT);
    digitalWrite(m_haptic_pin, LOW);
}

bool MarkusPTT::update() {
    bool reading = (digitalRead(m_ptt_pin) == LOW); // Active LOW

    if (reading != m_last_state) {
        m_last_debounce_time = millis();
    }

    if ((millis() - m_last_debounce_time) > m_debounce_delay_ms) {
        if (reading != m_debounced_state) {
            m_debounced_state = reading;
            if (m_debounced_state) {
                // Haptic tactile click on press
                pulseHaptic(60);
            }
        }
    }

    m_last_state = reading;
    return m_debounced_state;
}

void MarkusPTT::pulseHaptic(uint32_t duration_ms) {
    digitalWrite(m_haptic_pin, HIGH);
    delay(duration_ms);
    digitalWrite(m_haptic_pin, LOW);
}
