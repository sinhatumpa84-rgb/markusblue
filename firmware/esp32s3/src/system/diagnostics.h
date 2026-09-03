#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <stdint.h>
#include <stddef.h>

struct AudioLatencyMetrics {
    uint32_t dma_capture_us;
    uint32_t stft_us;
    uint32_t ai_inference_us;
    uint32_t istft_us;
    uint32_t post_dsp_us;
    uint32_t dma_tx_us;
    uint32_t total_pipeline_us;
    float total_latency_ms;
    uint32_t dropped_frames;
};

class MarkusDiagnostics {
private:
    AudioLatencyMetrics m_metrics;

public:
    MarkusDiagnostics();
    void recordFrameMetrics(
        uint32_t dma_rx,
        uint32_t stft,
        uint32_t ai,
        uint32_t istft,
        uint32_t dsp,
        uint32_t dma_tx
    );
    void recordDroppedFrame() { m_metrics.dropped_frames++; }
    const AudioLatencyMetrics& getMetrics() const { return m_metrics; }
    void printReport();
};

#endif // DIAGNOSTICS_H
