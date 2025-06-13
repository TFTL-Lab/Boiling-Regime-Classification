# Boiling-Regime-Classification

repository uses a fine-tuned version of **YAMNet** for **boiling regime classification**. The model takes an acoutic file as input and outputs the predicted regimes class based on learned features.

📁 repo Structure
.
├── get_yamnet_output.py       # Main script to run prediction
├── models/
│   └── yamnet_ndata.h5        # Fine-tuned YAMNet weights
├── yamnet_class_map_1.csv     # Class label map (CSV)
├── features.py                # Feature extraction functions
├── yamnet.py                  # YAMNet model definition
├── params.py                  # Model parameter definitions
└── README.md                  # Project documentation

⚙️ Requirements

Install dependencies with:

pip install -r requirements.txt

▶️ Usage

python3 get_yamnet_output.py path/to/audio.wav

📦 Output

The script will print:

Class label (from yamnet_class_map_1.csv)

Prediction probabilities

Example:

Predicted label: BKG
Probabilities : [0.76 0.01 0.22]


📑 Related Paper

This work is based on the following paper: Explainable Boiling Acoustics Analysis using Grad-CAM and YAMNet for Robust Pool Boiling Regime Classification
Suriyaprasaad B, Avinash Upadhyay, Atul Thakur, , and Rishi Raj
🔗 View Paper (URL)
📊 Dataset Source

The dataset used in this work is sourced from the following publication: Deep learning the sound of boiling for  advance prediction of boiling crisis 
Kumar Nishant Ranjan Sinha,  VijayKumar,NirbhayKumar, AtulThakur,RishiRaj
🔗 View Dataset Source Paper (URL)

🙏 Acknowledgements

TensorFlow YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

---

### ✅ What You Need to Add

Just replace the placeholders:

- `"Explainable Boiling Acoustics Analysis using Grad-CAM and YAMNet for Robust Pool Boiling Regime Classification"` → the title of **your paper**.
- `"https://paper-link.com"` → the **link to your paper** (arXiv, IEEE, etc.).
- `"Deeplearningthesoundofboilingfor
 advancepredictionofboilingcrisis KumarNishantRanjanSinha"` → the **title of the paper that released the dataset**.
- `"https://dataset-source-link.com"` → the **URL to the dataset paper** (DOI, publisher link, etc.).

