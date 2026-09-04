# MARKUSBLUE (SIH26052) — Army & DRDO Demonstration Plan

## 1. Demonstration Objective
To demonstrate to Indian Army / DRDO evaluators that MARKUSBLUE:
1. Enhances voice clarity in extreme battlefield noise environments on an embedded **ESP32-S3** microcontroller without cloud connectivity.
2. Preserves critical tactical cues (radio speech, alarms, sirens, footsteps) rather than indiscriminately muting all loud audio.
3. Provides hearing protection against sudden blast impulses with zero mute dropouts.

---

## 2. Stage-by-Stage Live Demonstration Protocol

### Test Station Setup
- **Workstation**: Laptop running `tools/sih_demo_suite.py` and serial monitor telemetry.
- **Audio Output**: Dual headphones / stereo speakers for judges to compare `RAW NOISY INPUT` vs. `MARKUSBLUE ENHANCED OUTPUT`.
- **Prototype Hardware**: ESP32-S3 N16R8 headset with OLED diagnostic display.

---

### Step 1: Baseline AI Speech Enhancement under Combat Vehicles (BMP-2 / T-90 Proxy)
- **Scenario**: Soldier speaking amidst 6-cylinder heavy diesel engine roar at 0 dB SNR.
- **Action**: Play raw noisy mixture, then activate MARKUSBLUE enhancement.
- **Evaluation Point**: Judges observe $>12\text{ dB}$ engine rumble attenuation while speech formants remain crisp and intelligible.

### Step 2: Critical Audio Preservation (Speech + Industrial Evacuation Alarm)
- **Scenario**: Heavy factory machinery running with a soldier shouting and a 1.1 kHz evacuation alarm sounding.
- **Action**: Run through MARKUSBLUE pipeline.
- **Evaluation Point**: Show that the machinery noise is attenuated while **both the voice and the alarm remain piercingly audible**, proving that MARKUSBLUE does not treat alarms as noise.

### Step 3: Tactical Radio Communication in Helicopter Cabin (Cheetah / ALH Proxy)
- **Scenario**: Narrowband tactical radio communication ("BRAVO-TWO-ZERO") with squelch bursts inside a helicopter cabin.
- **Action**: Stream audio through neural mask estimator.
- **Evaluation Point**: 16.7 Hz rotor slap is suppressed; radio voice and squelch clicks remain completely intelligible.

### Step 4: Blast Impulse Anti-Blanking Test
- **Scenario**: Soldier speaking when a high-caliber gunshot / explosion occurs.
- **Action**: Inject +12 dB transient spike.
- **Evaluation Point**: The peak limiter clamps the transient to safe levels ($\le -0.5\text{ dBFS}$) and **speech continues immediately without muting**, contrasting with commercial ear muffs that mute for 500 ms.

### Step 5: Fail-Safe Safe Bypass Mode
- **Scenario**: Deliberately disconnect or corrupt the neural inference task.
- **Action**: Trigger safe bypass mode.
- **Evaluation Point**: Audio transitions instantaneously to clean linear pass-through with peak limiting, ensuring the soldier is never left in silence.
