#ifndef SD_LOGGER_H
#define SD_LOGGER_H

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

class MarkusSDLogger {
private:
    bool m_card_mounted;
    int m_cs_pin;
    File m_log_file;
    bool m_logging_enabled;

public:
    MarkusSDLogger(int cs_pin = 10);
    bool init(int sck = 12, int miso = 13, int mosi = 11);
    void startRecording(const char* session_name);
    void stopRecording();
    void logTelemetry(float latency_ms, float snr_db, bool vad_active, int batt_pct);
    bool isMounted() const { return m_card_mounted; }
    bool isRecording() const { return m_logging_enabled; }
};

#endif // SD_LOGGER_H
