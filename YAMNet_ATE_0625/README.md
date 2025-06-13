# Boiling-Regime-Classification

This repository uses a fine-tuned version of **YAMNet** for **boiling regime classification**. The model takes an acoutic file as input and outputs the predicted regimes class based on learned features.

### 📁 Repository Structure
```
├── get_yamnet_output.py         # Main script to run prediction
├── models/
│   └── yamnet_ndata.h5          # Fine-tuned YAMNet weights
├── yamnet_class_map_1.csv       # Class label map (CSV)
├── features.py                  # Feature extraction functions
├── yamnet.py                    # YAMNet model definition
├── params.py                    # Model parameter definitions
└── README.md                    # Project documentation
```
---

### ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
### ▶️ Usage
```bash
python3 get_yamnet_output.py path/to/audio.wav
```
### 📦 Output

The script will print:

Class label (from yamnet_class_map_1.csv)

Prediction probabilities

Example:
``` bash
Predicted label: BKG
Prediction Probabilities : [0.76 0.01 0.22]
```

### 📚 References

1.  Suriyaprasaad, B., Upadhyay, A., Thakur, A., & Raj, R. (2025). **Explainable Boiling Acoustics Analysis using Grad-CAM and YAMNet for Robust Pool Boiling Regime Classification**. Link to Paper

2.  Sinha, K. N. R., Kumar, V., Kumar, N., Thakur, A., & Raj, R. (2021). **[Deep learning the sound of boiling for advance prediction of boiling crisis] (https://doi.org/10.1016/j.xcrp.2021.100382)**. Cell Reports Physical Science, 2(3).
  
3.  Sinha, K. N. R., Kumar, V., Kumar, N., Thakur, A., & Raj, R. (2024). **[Dataset for boiling acoustic emissions: A tool for data driven boiling regime prediction](https://doi.org/10.1016/j.dib.2023.109793)**. Data in Brief, 52, 109793.


### 🙏 Acknowledgements

TensorFlow YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
