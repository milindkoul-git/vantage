# Vantage: Architectural Case Study & Empirical Engineering Lessons

> *"The theory was written as fact, then measured to be false."*

This document pulls together five core architectural lessons learned during the design, construction, and benchmarking of Vantage (v0.11.0). Each case illustrates a recurring theme: how textbook computer vision and data engineering assumptions fail in production, and how empirical measurement drove the architecture.

---

## 1. The Kalman Size-Extrapolation Trap in Multi-Object Tracking

### The Textbook Assumption
Standard 2D tracking filters (such as standard SORT or DeepSORT baselines) model the state vector as $[x_c, y_c, w, h, \dot{x}_c, \dot{y}_c, \dot{w}, \dot{h}]^T$, applying a linear constant-velocity motion model to both box position and box dimensions.

### The Real-World Failure
In video analytics, objects frequently enter partial occlusions (e.g., a person walking behind a pillar or parking behind a barrier). As the detector encounters the obstacle, the visible bounding box rapidly shrinks in width over 2–3 frames. 

Because box width was modeled with velocity $\dot{w}$, the Kalman filter observed this rapid shrinkage as a strong negative size derivative. When the object disappeared completely into total occlusion and the tracker began coasting:
1. The filter extrapolated $\dot{w} < 0$, shrinking the predicted bounding box down to zero width or inverted coordinates within fractions of a second.
2. When the person emerged from behind the pillar 1.2 seconds later, the predicted box had collapsed to a sliver.
3. Spatial IoU between the newly detected box and the collapsed prediction dropped to `0.00`, causing ByteTrack to fail association and issue a completely new `entity_id`.

### The Vantage Solution
**Position is extrapolated; size deliberately is not.**
Box dimensions $(w, h)$ are updated as random walks with a bounded drift noise ($\sigma_{\text{drift}} = 0.2$), but maintain zero velocity derivative ($\dot{w} = 0, \dot{h} = 0$). Predicted boxes during coasting retain their last known observed aspect ratio and dimensions, allowing identity to reliably survive up to 1.5 seconds of total visual occlusion without an appearance model.

---

## 2. Statistical Baseline Collapse: Median & MAD vs. Mean & StdDev

### The Textbook Assumption
Anomaly detection on traffic and occupancy is traditionally framed as:
$$\text{Anomaly} \iff x_t > \mu_{\text{slot}} + k \cdot \sigma_{\text{slot}}$$

### The Catastrophic Failure Mode
Gaussian parameters have a breakdown point of $0\%$: a single extreme outlier completely dominates both $\mu$ and $\sigma$.

Consider an office corridor that averages 3 people/hour on Monday mornings ($\mu = 3.0, \sigma = 1.2$). One Monday, an all-hands meeting causes 45 people to walk through in an hour:
- With Mean & StdDev, the updated baseline shifts to $\mu = 13.5, \sigma = 18.2$.
- The anomaly threshold $\mu + 3\sigma$ inflates from $6.6$ to **$68.1$**.
- Having witnessed one unusual event, the detector has permanently taught itself that massive crowds are ordinary. Future crowds of 30–40 people will never be flagged again.

### The Vantage Solution
Vantage fits baselines using the **Median** as the center and the **Median Absolute Deviation (MAD)** as the spread:
$$\text{MAD} = \text{median}(|x_i - \text{median}(X)|)$$
$$\sigma_{\text{est}} = 1.4826 \times \text{MAD}$$

The median and MAD tolerate up to **50% contaminated data** before shifting. A spike of 45 people leaves the learned center at 3.0, guaranteeing that subsequent anomalies remain detectable.

---

## 3. The Synthetic Testing Illusion: Deterministic Pass vs. Real Tails

### The Discovery
During Phase 11 validation, the scenario test suite passed with 100% accuracy, reporting **0.00 false alarms per week**. 

However, running the statistical detector against realistic traffic distributions with natural variance revealed that the system was actually producing **~10 false alarms per week**.

### Why the Unit Tests Lied
The automated scenario harness used deterministic, uniformly bounded jitter ($\pm 10\%$). Real human arrival counts follow Poisson and Negative Binomial distributions with long tails:
1. With only 4 weeks of history, each time slot has only 4 samples.
2. The MAD of 4 clustered points is biased low by a factor of $2.7\times$.
3. What was labeled as a $3.5\sigma$ threshold was mathematically functioning as a $2.3\sigma$ threshold.

### The Fix: `vantage analytics characterise`
1. Vantage implemented **shrinkage towards a pooled variance estimate**: quiet slots with few samples are regularized against the camera's overall residual variance.
2. A Monte-Carlo characterization harness (`vantage analytics characterise`) was built into the CLI to continuously sample heavy-tailed distributions and quantify exact false positive rates ($0.09/\text{week}$ on clean data).

---

## 4. Privacy by Architecture, Not Policy

### The Requirement
Video analytics must provide spatial and behavioral understanding without compromising individual privacy or creating biometric surveillance liabilities.

### The Vantage Architecture
1. **Anonymous Tracking by Default**: ByteTrack operates purely on 2D bounding-box geometry without extracting appearance embeddings.
2. **Explicit Consent Gate**: The facial identification engine is disabled by default and requires the `--consent` flag during enrollment.
3. **No Image Storage**: Raw facial crops are never persisted to disk or cached. Only irreversible 128-dimensional floating point vectors are stored in an isolated database.
4. **AST-Enforced Separation**: Unit tests parse the Python Abstract Syntax Tree (AST) of the tracking and live pipeline modules, asserting that no live pipeline stage can import the enrollment module or call gallery write methods.

---

## 5. Dynamic Graph Compilation on Intel iGPUs

### The Measurement
When evaluating OpenVINO execution on Intel Iris Xe / Arc iGPUs:
- Loading a model with **dynamic input shapes** forced the OpenVINO driver to re-evaluate kernel graphs at runtime, resulting in **184 ms/frame** inference latency.
- Explicitly pinning and **reshaping the input tensor** before `compile_model` brought latency down to **84 ms/frame** ($2.2\times$ speedup).
- In contrast, one-shot open-vocabulary discovery (Grounding DINO) spent 155 seconds compiling the GPU graph to save 9 seconds of runtime, proving that **CPU execution is 7× faster than GPU for one-shot zero-shot tasks**.

---

## Summary

The core philosophy underlying Vantage is that **verifiable, auditable evidence on standard hardware beats opaque, confident predictions**. Every threshold, coordinate convention, and state transition in the system produces structured explanatory evidence that human operators can verify.
