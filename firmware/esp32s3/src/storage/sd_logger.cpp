#include "sd_logger.h"

MarkusSDLogger::MarkusSDLogger(int cs_pin)
    : m_card_mounted(false),
      m_cs_pin(cs_pin),
      m_logging_enabled(false) {}

bool MarkusSDLogger::init(int sck, int miso, int mosi) {
    SPI.begin(sck, miso, mosi, m_cs_pin);
    if (!SD.begin(m_cs_pin, SPI, 20000000)) { // 20 MHz SPI
        m_card_mounted = false;
        return false;
    }
    m_card_mounted = true;
    return true;
}

void MarkusSDLogger::startRecording(const char* session_name) {
    if (!m_card_mounted) return;

    char filename[64];
    snprintf(filename, sizeof(filename), "/log_%s.csv", session_name);
    m_log_file = SD.open(filename, FILE_WRITE);
    if (m_log_file) {
        m_log_file.println("timestamp_ms,latency_ms,snr_db,vad_active,battery_pct");
        m_log_file.flush();
        m_logging_enabled = true;
    }
}

void MarkusSDLogger::stopRecording() {
    if (m_log_file) {
        m_log_file.flush();
        m_log_file.close();
    }
    m_logging_enabled = false;
}

void MarkusSDLogger::logTelemetry(float latency_ms, float snr_db, bool vad_active, int batt_pct) {
    if (!m_logging_enabled || !m_log_file) return;

    m_log_file.printf("%lu,%.2f,%.2f,%d,%d\n", millis(), latency_ms, snr_db, vad_active ? 1 : 0, batt_pct);
}
