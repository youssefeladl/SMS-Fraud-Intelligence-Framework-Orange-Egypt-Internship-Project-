Uncovering the Invisible: Building a Telecom-Scale Fraud Intelligence Framework

At Orange Egypt – Big Data & AI, I had the privilege to architect and deliver a system that redefines SMS fraud detection—elevating it from a single algorithm into a multi-layered fraud intelligence framework.
<img width="1536" height="1024" alt="98831fe3-e670-4f1f-a6a2-20b7e7b139a8" src="https://github.com/user-attachments/assets/da2c7f3c-73b9-43a6-b526-2b2f5bc9dc31" />
This was never about “running a model.” It was about designing a scientific investigation pipeline where every tool—from EDA to anomaly detection to supervised learning—became part of one coherent shield.

🔬 1. Exploratory Data Analysis as an Instrument
EDA wasn’t a checkbox. It was an investigative weapon:

Performed quantile drilling at 5% steps, uncovering subtle irregularities invisible at the surface.

Identified 0.1% (≈0.001) of traffic as the empirically valid anomaly threshold—rare enough to be statistically significant, critical enough to be operationally meaningful.

This quantile-based insight wasn’t just analysis, it was the compass that guided every downstream decision.

🕵 2. Outlier Discovery → Structured Knowledge

Launched with Isolation Forest to uncover latent anomalies.

Applied clustering not for its end labels, but as a lens—a diagnostic instrument to reveal hidden groupings and validate anomaly structure.

Translated unsupervised signals into a labeled fraud dataset, bridging the gap between raw exploration and supervised modeling.

🌲 3. Modeling with Purpose

Built a Random Forest classifier informed by anomaly labels.

Optimized deliberately for high recall: in fraud, catching all threats comes before narrowing precision.

Adjusted contamination ratios using outlier-driven calibration from the EDA phase—blending statistical rigor with business imperatives.

⚙ 4. Operational Deployment

Delivered a production-grade Streamlit platform:

Capable of scoring multi-million row datasets (CSV/Parquet/ZIP) seamlessly.

Aggregates anomalies by sender, quantifies impact, and prioritizes suspicious entities.

Generates downloadable anomaly reports and interactive analytics for real-time decision-making.

Not a prototype, but a scalable operator-ready tool

app

.

🎯 Core Realization
Fraud detection is not the art of chasing a “perfect algorithm.”
It is the science of layering methods, instruments, and thresholds into a framework that adapts to the scale and complexity of real-world systems.

🙏 Acknowledgements
Grateful to Kareem Seliman and the Orange Big Data & AI team—your mentorship and sharp feedback transformed this project into an applied research milestone.

🔸 Mastered domains: Quantile-Based Outlier Mining · EDA as Instrument · Isolation Forest · Clustering · Random Forest · High-Recall Modeling · Big Data Deployment
🔸 Mission: advancing the frontier where AI safeguards telecom ecosystems at scale.

#FraudIntelligence #AIinTelecom #OrangeEgypt #BigData #AnomalyDetection #MachineLearning
