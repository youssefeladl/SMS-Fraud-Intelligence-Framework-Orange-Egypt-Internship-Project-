# 🚨 SMS Fraud Intelligence Framework  
**Orange Egypt – Big Data & AI Internship Project**  
<img width="1536" height="1024" alt="98831fe3-e670-4f1f-a6a2-20b7e7b139a8" src="https://github.com/user-attachments/assets/da2c7f3c-73b9-43a6-b526-2b2f5bc9dc31" />
https://qyhnfssnsdrcvb32huxrlx.streamlit.app/

An end-to-end system for detecting and analyzing fraudulent SMS traffic at telecom scale.  
This project combines exploratory analysis, anomaly detection, supervised learning, and a production-ready dashboard — designed not just as a prototype, but as an **operator-level fraud intelligence tool**.  
---
## 📌 Overview  

Billions of SMS messages flow through telecom networks daily. Hidden within them lies a fraction of traffic that represents fraud — small in volume, but significant in impact.  

This framework was built to uncover those invisible patterns using a **multi-layered approach**:  

- **Quantile-based Exploratory Data Analysis (EDA)** to identify anomaly thresholds.  
- **Isolation Forest** for unsupervised outlier discovery.  
- **Clustering** as a diagnostic instrument to validate anomaly structures.  
- **Random Forest** classifier trained on labeled anomalies, optimized for high recall.  
- **Streamlit dashboard** to operationalize detection with scalability and usability.  

---

## 🔬 Methodology  

1. **EDA as Foundation**  
   - Performed quantile drilling (every 5%) across millions of SMS logs.  
   - Determined 0.1% (≈0.001) as the logical anomaly ratio.  

2. **Anomaly Discovery & Labeling**  
   - Detected anomalies via Isolation Forest.  
   - Used clustering to interpret anomaly distribution.  
   - Converted signals into labeled fraud datasets.  

3. **Modeling**  
   - Trained a Random Forest classifier for supervised detection.  
   - Prioritized **high recall** to minimize missed fraud cases.  

4. **Deployment**  
   - Built a Streamlit app with:  
     - Multi-format support (CSV, Parquet, ZIP).  
     - Sender-level aggregation & ranking.  
     - Exportable anomaly reports.  
     - Interactive charts for anomaly distributions.  

---

## ⚡ Features  

- Scalable scoring for **multi-million row datasets**.  
- Hybrid fraud detection: **Isolation Forest + Random Forest**.  
- Operator-grade interface with **Streamlit**.  
- Exportable anomaly insights for investigation teams.  

---

## 🛠️ Tech Stack  

- **Python** (Pandas, NumPy, scikit-learn)  
- **Isolation Forest & Random Forest**  
- **Clustering (EDA tool)**  
- **Matplotlib** (visual analytics)  
- **Streamlit** (dashboard)  

---

## 🙏 Acknowledgements  

Special thanks to **Kareem Seliman** and the Orange Big Data & AI team for their mentorship and valuable feedback throughout this project.  

---

## 🎯 Mission  

To advance the frontier where **AI safeguards telecom ecosystems** by turning raw SMS data into real-time fraud intelligence.  
