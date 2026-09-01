# MARKUSBLUE Model Audit Test Summary

## Test Results Across 8 Operational Scenarios

| Scenario | In SI-SDR | Out SI-SDR | Gain | In STOI | Out STOI | In RMS | Out RMS | Speech Loudness Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Clean Speech | 120.18 dB | 10.84 dB | **+-109.34 dB** | 1.0 | **0.981** | 0.1146 | 0.0814 | ATTENUATED |
| Speech + Background Noise (+5 dB) | 4.97 dB | 8.88 dB | **+3.91 dB** | 0.935 | **0.97** | 0.1274 | 0.1106 | AUDIBLE (Loudness Maintained) |
| Speech + Background Noise (-5 dB Low SNR) | -5.07 dB | 1.94 dB | **+7.02 dB** | 0.744 | **0.891** | 0.2336 | 0.1344 | AUDIBLE (Loudness Maintained) |
| Speech + Gunshot Impulse (+0 dB) | -0.07 dB | 9.75 dB | **+9.81 dB** | 0.852 | **0.975** | 0.1504 | 0.1346 | AUDIBLE (Loudness Maintained) |
| Speech + Gunshot Impulse (-10 dB Heavy) | -9.97 dB | 6.83 dB | **+16.79 dB** | 0.651 | **0.955** | 0.3953 | 0.1353 | AUDIBLE (Loudness Maintained) |
| Speech + Other Impulse (Machinery/Impact) | 0.0 dB | 12.13 dB | **+12.12 dB** | 0.854 | **0.985** | 0.1724 | 0.1559 | AUDIBLE (Loudness Maintained) |
| Very Noisy Mixture (-15 dB Extreme) | -15.31 dB | -3.33 dB | **+11.97 dB** | 0.585 | **0.782** | 0.6849 | 0.1048 | AUDIBLE (Loudness Maintained) |
| Speech Loudness Test (Low Volume Input) | 10.07 dB | 9.43 dB | **+-0.64 dB** | 0.977 | **0.974** | 0.0175 | 0.0184 | ATTENUATED |
