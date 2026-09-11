### Marker Quality Cascade — Computation

**START: Is the marker detected?**
→ **NO:** `MISSING`
→ **YES:** continue

↓

### 1. FOREARM: pairwise distance check

For every forearm-marker pair:

* Compute the **frame-by-frame distance**
* Compare it with a **301-frame rolling median + MAD**
* Reject the pair if the distance deviation exceeds
  **max(4 × local MAD, 2.5 mm)**
* Reject the local baseline if it differs from the global reference by **>10 mm**

**Are ≥2 forearm markers flagged?**
→ **YES:** **FRAME VETO → all present markers = INCORRECT**
→ **NO:** continue

↓

### 2. ANCHOR: distance to forearm centroid

* Compute the **forearm centroid**
* For each Palm/finger marker, compute its **distance to the centroid**
* Compare with the reference-frame distribution
* Flag the marker if deviation exceeds
  **max(1.5 × MAD, 6 mm)**

↓

### 3. PALM: 2/2/2 quorum

Group = **Palm 1 + Palm 2 + Palm 3 + Thumb1**

Count three things:

* **≥2 markers present**
* **≥2 pass bone-length check**
* **≥2 pass anchor check**

**Does all three minimum counts hold?**
→ **NO:** **PALM + FINGERS = INCORRECT**
→ **YES:** continue

↓

### 4. FINGERS: sequential distance checks

**Finger base:**

* Compare base with each correct Palm marker
* Also require the base to pass its **anchor check**

**Next joint:**

* Compute distance to the **previous marker**
* Compare against the bone-length threshold

**Does the link pass?**
→ **NO:** this marker **AND all distal markers = INCORRECT**
→ **YES:** move to the next joint

↓

### 5. TEMPORAL CHECK

For each marker:

* Compute displacement from its **previous detected position**
* Divide by elapsed frames → **movement speed**
* Flag if speed exceeds
  **max(4 × speed MAD, 12.5 mm/frame)**

→ **Exceeded:** `INCORRECT`
→ **Not exceeded:** continue

↓

### 6. FINAL OVERRIDES / RECOVERY

**Is this a verified reference frame?**
→ **YES:** `CORRECT`
→ **NO:** continue

**Is the marker only unverified/uncertain, rather than measured as bad?**
→ **NO:** keep `INCORRECT`
→ **YES:** continue

**Is it ≤4 mm from a trusted neighbouring frame?**
→ **YES:** **RECOVER → CORRECT**
→ **NO:** keep `INCORRECT`

### Overall logic

**Measure distances → compare with thresholds → apply upstream gates → propagate failures along the chain → apply temporal check → apply reference/recovery rules.**
