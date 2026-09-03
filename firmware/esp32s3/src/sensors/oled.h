#ifndef OLED_H
#define OLED_H

#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_I2C_ADDR 0x3C
#define OLED_RESET_PIN -1

struct SystemDisplayState {
    bool ai_active;
    bool mic_ok;
    bool enhancement_on;
    bool recording_active;
    int battery_pct;
    float latency_ms;
    float snr_db;
    bool ptt_pressed;
};

class MarkusDisplay {
private:
    Adafruit_SSD1306 m_display;
    bool m_initialized;

public:
    MarkusDisplay();
    bool init(int sda_pin = 8, int scl_pin = 9);
    void render(const SystemDisplayState& state);
    void showBootLogo();
    void showError(const char* msg);
};

#endif // OLED_H
