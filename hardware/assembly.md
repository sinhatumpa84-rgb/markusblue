# MARKUSBLUE (SIH26052) — Physical Prototype Assembly & Mechanical Integration Guide

## 1. Mechanical Overview & Ear-Cup Enclosure
The MARKUSBLUE tactical audio prototype utilizes a rugged circumaural dual-cup headset design.
- **Right Ear-Cup**: Hosts the primary electronics stack (ESP32-S3 DevKit, power distribution PCB, TP4056 charger, and MAX98357A amplifier).
- **Left Ear-Cup**: Houses the 3.7V 2500mAh Li-Po battery reservoir to achieve symmetric weight distribution across the operator's head.
- **Outer Shell Mount**: Hosts External Reference INMP441 (Mic 1) protected by stainless steel mesh and hydrophobic acoustic membrane.
- **Inner Cavity Mount**: Hosts Internal Ear-Side INMP441 (Mic 2) positioned 5mm from the 40mm speaker baffle.

---

## 2. Step-by-Step Electrical & Mechanical Assembly

### Step 1: Acoustic Chamber Preparation
1. Mount the 40mm 8Ω 2W speaker transducer onto the internal ear-cup baffle using a silicone vibration-dampening ring. Ensure airtight seal around the transducer perimeter.
2. Position the **Internal INMP441 (Mic 2)** directly adjacent to the speaker baffle facing the operator's ear canal. Solder `L/R` to `GND` (Right Channel).
3. Fill the rear ear-cup cavity with 15mm open-cell acoustic absorption foam to absorb rear speaker back-waves and prevent acoustic standing waves.

### Step 2: External Reference Microphone Installation
1. Drill a 2.5mm acoustic port on the forward-facing outer shell of the right ear-cup.
2. Install a water-repellent acoustic fabric (e.g. SaatiChem Acoustex) over the port.
3. Position the **External INMP441 (Mic 1)** behind the acoustic port. Solder `L/R` to `VDD (3.3V)` (Left Channel).
4. Connect shared BCLK (GPIO 4), shared WS (GPIO 5), and shared SD (GPIO 6) using shielded twisted-pair wire.

### Step 3: Electronics PCB & Amplifier Mounting
1. Secure the ESP32-S3 board onto the mounting standoffs using M2 nylon screws.
2. Mount the MAX98357A amplifier within 30mm of the speaker terminals to minimize EMI radiation from Class-D switching lines.
3. Solder a 220 µF low-ESR electrolytic capacitor across the 5V power supply input of the MAX98357A.
4. Wire I2S1 lines (BCLK: GPIO 15, WS: GPIO 16, DIN: GPIO 17) with ground trace shielding.

### Step 4: Battery & Power Routing
1. Install the 2500mAh Li-Po cell into the left ear-cup with EVA shock-absorbing foam.
2. Route a 2-conductor 22 AWG power harness through the padded headband to the right ear-cup power switch.
3. Connect through the TP4056 protection board with an inline master power toggle switch.
4. Mount the USB-C charging port on the lower rim of the right ear-cup with a rubber dust plug.

### Step 5: Peripheral & UI Integration
1. Mount the 0.96" OLED display on the top-outer bezel of the right ear-cup for easy diagnostic viewing.
2. Position the PTT tactical pushbutton on the lower outer rim where it can be actuated with gloved fingers.
3. Mount the 1027 coin haptic motor on the inner headband cushion directly contacting the operator's skull/mastoid process for silent tactile alerts.
