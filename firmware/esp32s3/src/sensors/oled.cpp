#include "oled.h"
#include <Wire.h>

MarkusDisplay::MarkusDisplay()
    : m_display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET_PIN),
      m_initialized(false) {}

bool MarkusDisplay::init(int sda_pin, int scl_pin) {
    Wire.begin(sda_pin, scl_pin, 400000); // 400 kHz Fast I2C
    if (!m_display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
        m_initialized = false;
        return false;
    }
    m_initialized = true;
    showBootLogo();
    return true;
}

void MarkusDisplay::showBootLogo() {
    if (!m_initialized) return;
    m_display.clearDisplay();
    m_display.setTextColor(SSD1306_WHITE);

    m_display.setTextSize(1);
    m_display.setCursor(18, 8);
    m_display.print("PROJECT MARKUSBLUE");

    m_display.drawLine(10, 20, 118, 20, SSD1306_WHITE);

    m_display.setCursor(20, 26);
    m_display.print("SIH26052 TACTICAL");
    m_display.setCursor(16, 38);
    m_display.print("EDGE-AI AUDIO ANC");

    m_display.drawLine(10, 50, 118, 50, SSD1306_WHITE);

    m_display.setCursor(26, 54);
    m_display.print("INITIALIZING...");
    m_display.display();
}

void MarkusDisplay::render(const SystemDisplayState& state) {
    if (!m_initialized) return;

    m_display.clearDisplay();
    m_display.setTextColor(SSD1306_WHITE);

    // Header bar
    m_display.setTextSize(1);
    m_display.setCursor(0, 0);
    m_display.print("MARKUSBLUE");

    // Battery percentage right aligned
    m_display.setCursor(82, 0);
    m_display.printf("BAT:%d%%", state.battery_pct);
    m_display.drawLine(0, 9, 127, 9, SSD1306_WHITE);

    // Row 1: AI & Microphone Status
    m_display.setCursor(0, 12);
    m_display.printf("AI: %s", state.ai_active ? "ACTIVE [ON]" : "STANDBY");

    m_display.setCursor(0, 22);
    m_display.printf("MIC: %s", state.mic_ok ? "DUAL I2S OK" : "MIC ERROR");

    // Row 2: Latency & SNR
    m_display.setCursor(0, 32);
    m_display.printf("LATENCY: %.1f ms", state.latency_ms);

    m_display.setCursor(0, 42);
    m_display.printf("SNR EST: %+.1f dB", state.snr_db);

    // Footer Bar: PTT & Recording Status
    m_display.drawLine(0, 52, 127, 52, SSD1306_WHITE);
    m_display.setCursor(0, 55);
    if (state.ptt_pressed) {
        m_display.print("PTT: TRANSMITTING");
    } else {
        m_display.printf("MODE: %s", state.recording_active ? "REC ACTIVE" : "ANC PASSIVE");
    }

    m_display.display();
}

void MarkusDisplay::showError(const char* msg) {
    if (!m_initialized) return;
    m_display.clearDisplay();
    m_display.setTextColor(SSD1306_WHITE);
    m_display.setTextSize(1);
    m_display.setCursor(0, 10);
    m_display.print("SYSTEM FAULT:");
    m_display.setCursor(0, 25);
    m_display.print(msg);
    m_display.display();
}
