#ifndef AUDIO_BUFFER_H
#define AUDIO_BUFFER_H

#include <stdint.h>
#include <stddef.h>

class AudioRingBuffer {
private:
    float* m_buffer;
    size_t m_capacity;
    size_t m_head;
    size_t m_tail;
    size_t m_count;

public:
    AudioRingBuffer(size_t capacity);
    ~AudioRingBuffer();

    bool write(const float* data, size_t len);
    bool read(float* data, size_t len);
    size_t available() const;
    size_t freeSpace() const;
    void clear();
};

#endif // AUDIO_BUFFER_H
