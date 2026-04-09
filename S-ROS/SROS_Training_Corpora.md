# S-ROS — Training Corpora & Telemetry Sources

*Genesis Labs Research · S-ROS Module · 2026*

---

## Intellectual Property Notice

S-ROS does not distribute, incorporate, or make use of any copyrighted material without explicit written permission from the copyright holder. S-ROS respects intellectual property and categorically will not use such materials without explicit permission. All datasets listed here are either U.S. Government public domain, released under CC0 or CC-BY licenses, or generated from open-source firmware and simulators.

If you would like to make your training set available to S-ROS, contact us via the Genesis Labs Research repository.

---

## 01 · Primary Deterministic Datasets

*U.S. Government · Public Domain*

### NIST RS274NGC Interpreter Samples

**Source:** NISTIR 6556 Full Report · CC-BY · Open

Official U.S. Government reference with sample G-code programs, canonical machining functions, and error cases. Core public-domain foundation for syntactic standards, lookup tables, and happy-path training in the Neuro-Symbolic Translator.

*Applications: Syntactic standards · Lookup table seeding · Happy-path training*

---

### CPS_HIDS G-code Dataset

**Source:** GitHub · NIST RS-274 / ISO 6983-1:2009 compliant · CC-BY 4.0

G-code dataset for 3D printers and CNC machines including normal operation and injected anomaly scenarios. Trains CA1 novelty mismatch detection and supervised anomaly reflexes.

*Applications: Anomaly detection · CA1 novelty training · Fault injection*

---

### Bosch CNC Machining Dataset

**Source:** UCI ML Repository · U.S. Government · Public Domain

High-frequency vibration and acceleration telemetry from real industrial CNC milling. Grounds physical anomaly detection — tool breakage, chatter, stalls — and STDP survival reflexes in empirical machine behavior.

*Applications: Physical anomaly detection · STDP survival reflexes · Telemetry grounding*

---

### ORNL Additive Manufacturing Datasets

**Source:** OSTI.gov · Oak Ridge National Laboratory · U.S. Government · Public Domain

Large-scale public datasets including in-situ sensor data, imaging, and mechanical testing for 3D-printed parts. Anchors quality assessment, layer adhesion, and physics-grounded training in real hardware outcomes.

*Applications: Quality assessment · Layer adhesion · Hardware outcome grounding*

---

### CNC Mill Tool Wear Dataset

**Source:** Kaggle · CC0 · Public Domain

Variational CNC machining data with tool condition, feed rate, and clamping pressure variations. Pairs finished parts with process data for tool wear and process monitoring reflexes grounded in real failure modes.

*Applications: Tool wear monitoring · Process failure modes · Reflex grounding*

---

## 02 · Firmware & Telemetry Generators

### Klipper & Marlin

**3D Printer Firmware · Open Source**

Generates raw spatial coordination (X/Y/Z/E) and extruder resistance logs during real-time operation.

---

### LinuxCNC & Machinekit

**Industrial Controller · Open Source**

Supplies precise multi-axis subtractive manufacturing logs and spindle telemetry from production environments.

---

### ROS2

**Robotics Middleware · Open Source**

Records and replays synchronized `geometry_msgs`, `sensor_msgs`, and `trajectory_msgs` from real or simulated hardware for reproducible scenario modeling.

---

## 03 · Syntactic Standards

### NIST RS274NGC Interpreter

**Version 3 · NISTIR 6556 · Public Domain**

The definitive vendor-neutral reference implementation for G-code and M-code. Builds fixed, unambiguous lookup tables in the Neuro-Symbolic Translator.

---

### CNC Programming Handbook

**Status: Pending — Seeking Permission**

Widely recognized industry reference for aligning mechanical primitives — Circular Interpolation, Dwell, Tool Compensation — with real-world CNC practice. We have not yet referenced or integrated this work. We are actively seeking formal permission from Industrial Press and Peter Smid before any use.

---

*Genesis Labs Research · S-ROS Training Infrastructure · 2026*
*Open, legally clean, high-precision neuro-symbolic intelligence for manufacturing.*
*github.com/genesislabs-research/PRAGMI-Architecture-/tree/main/S-ROS*
