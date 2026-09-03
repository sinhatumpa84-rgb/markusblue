#include "audio_buffer.h"
#include <esp_heap_caps.h>
#include <string.h>

AudioRingBuffer::AudioRingBuffer(size_t capacity)
    : m_capacity(capacity), m_head(0), m_tail(0), m_count(0) {
    // Allocate buffer in internal SRAM (DMA-accessible) for minimal latency
    m_buffer = (float*)heap_caps_malloc(capacity * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!m_buffer) {
        // Fallback to PSRAM if large
        m_buffer = (float*)heap_caps_malloc(capacity * sizeof(float), MALLOC_CAP_SPIRAM);
    }
    if (m_buffer) {
        memset(m_buffer, 0, capacity * sizeof(float));
    }
}

AudioRingBuffer::~AudioRingBuffer() {
    if (m_buffer) {
        free(m_buffer);
        m_buffer = nullptr;
    }
}

bool AudioRingBuffer::write(const float* data, size_t len) {
    if (!m_buffer || len > freeSpace()) return false;

    for (size_t i = 0; i < len; ++i) {
        m_buffer[m_head] = data[i];
        m_head = (m_head + 1) % m_capacity;
    }
    m_count += len;
    return true;
}

bool AudioRingBuffer::read(float* data, size_t len) {
    if (!m_buffer || len > m_count) return false;

    for (size_t i = 0; i < len; ++i) {
        data[i] = m_buffer[m_tail];
        m_tail = (m_tail + 1) % m_capacity;
    }
    m_count -= len;
    return true;
}

size_t AudioRingBuffer::available() const {
    return m_count;
}

size_t AudioRingBuffer::freeSpace() const {
    return m_capacity - m_count;
}

void AudioRingBuffer::clear() {
    m_head = 0;
    m_tail = 0;
    m_count = 0;
    if (m_buffer) {
        memset(m_buffer, 0, m_capacity * sizeof(float));
    }
}
