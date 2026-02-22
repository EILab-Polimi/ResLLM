# Reservoir Configuration File – README

This document explains the purpose and physical meaning of each parameter in the reservoir configuration file. The values and relationships defined here are used by the model to represent the real-world operational, hydraulic, and flood-control behavior of a large reservoir (e.g., Folsom Lake).

---

## operable_storage_max: 975 (TAF)

**Meaning:**  
The maximum volume of water the reservoir can hold for normal operations. This represents the modeled “full pool.”

**Context:**  
The physical capacity of Folsom Lake is approximately 977 TAF, so 975 TAF is used as a practical operational maximum.

**Unit:**  
TAF = Thousand Acre-Feet

**Why it matters:**  
The model treats this as the upper bound of usable storage. Water above this level is considered spill or emergency storage.

---

## operable_storage_min: 90 (TAF)

**Meaning:**  
The minimum operable storage, also known as the *dead pool*.

**Context:**  
Water below this level cannot be reliably released through gravity-fed outlets or is reserved for sediment control and ecosystem survival.

**Why it matters:**  
This defines the lower bound of usable storage. Below this level, the reservoir is effectively non-operational.

---

## max_safe_release: 130000 (cfs)

**Meaning:**  
The maximum safe release rate into the downstream river channel.

**Context:**  
Releases greater than 130,000 cubic feet per second (cfs) into the American River would likely cause levee failure and catastrophic flooding in Sacramento.

**Model Behavior:**  
This is treated as a hard constraint:
- The model will *never* release more than this value during normal operations.
- Only emergency spill logic may exceed this limit.

**Unit:**  
cfs = Cubic Feet per Second

---

## sp_to_ep (Storage → Elevation Relationship)

**Description:**  
Maps reservoir storage volume to water surface elevation.

**Input (Top List):**  
Storage (TAF)

**Output (Bottom List):**  
Elevation (feet above sea level)

**Purpose:**  
This curve represents the bathymetry (shape) of the reservoir basin.

**Why it matters:**  
As storage increases, the water surface elevation rises. Higher elevation creates more hydraulic head, which:
- Increases hydroelectric generation potential
- Increases the ability to push water through outlets at higher rates

---

## tp_to_tocs (Time → TOCS Rule Curve)

**Description:**  
Maps time of year to the maximum allowed storage.

**Input (Top List):**  
Day of Water Year (0–365)

**Note:**  
The water year typically starts on October 1st, so:
- Day 0 ≈ October 1
- Day 365 ≈ September 30

**Output (Bottom List):**  
TOCS (Top of Conservation Storage) in TAF

**Purpose:**  
Defines the flood control rule curve.

**Seasonal Behavior:**
- **Winter (approximately day 50–150):**  
  TOCS drops to ~400 TAF, leaving ~575 TAF of empty space for flood capture.
- **Summer:**  
  TOCS rises back to 975 TAF to maximize water supply storage.

**Why it matters:**  
This prevents overfilling during flood season while allowing full storage during dry months.

---

## sp_to_rp (Storage → Release Capacity Relationship)

**Description:**  
Maps reservoir storage to the maximum physically possible release rate.

**Input (Top List):**  
Storage (TAF)

**Output (Bottom List):**  
Maximum release (cfs)

**Purpose:**  
Represents the hydraulic limits of the dam’s outlet works.

**Behavior:**
- At very low storage (near 90 TAF), water pressure is insufficient, and releases may be zero.
- At higher storage levels (e.g., 400 TAF and above), sufficient pressure exists to reach maximum outlet capacity (e.g., ~40,000 cfs).

**Why it matters:**  
Even if policy allows a large release, the dam cannot exceed these physical limits.

---

## Unit and Term Reference

| Parameter | Full Name                     | Unit                  | Meaning |
|---------|--------------------------------|-----------------------|---------|
| TAF     | Thousand Acre-Feet             | Volume                | Large-scale reservoir volume |
| cfs     | Cubic Feet per Second          | Flow                  | Rate of water movement |
| TOCS    | Top of Conservation Storage    | Volume                | Target storage limit to preserve flood space |
| sp_to_ep| Storage to Elevation           | Curve                 | If the lake has V water, the surface is at H height |
| tp_to_tocs | Time to TOCS               | Curve                 | How much storage is allowed on a given day |

---

## Additional Notes

- **1 Acre-Foot** is the volume of water needed to cover one acre of land to a depth of one foot.  
  This is approximately **325,851 gallons**.

This configuration ensures that the model respects real-world physical, hydraulic, and flood-control constraints while simulating reservoir operations.
