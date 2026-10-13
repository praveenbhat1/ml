# AgriTech Intelligence: Crop Yield Predictor

![AgriTech Intelligence Banner](https://img.shields.io/badge/AgriTech-Intelligence-1D9E75?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

AgriTech Intelligence is a machine learning demo that classifies crop yield potential (High/Low). By analyzing environmental factors, cultivation data, and historical trends, the platform provides actionable insights for farmers and agricultural researchers.

## 🚀 Features

- **Real-time Classification**: Instant yield analysis (High/Low) using a pre-trained Random Forest model.
- **Inputs**: Crop, Season, State, Year, Area, Production, Rainfall, Fertilizer and Pesticide.
- **Premium UI**: Sleek, dark-themed responsive interface built for modern browsers.
- **Cross-Platform**: Works seamlessly on desktop and mobile devices.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Random Forest, Pandas, Numpy, Scikit-learn, Joblib
- **Frontend**: HTML5, Vanilla CSS3 (Manrope & Space Grotesk typography)
- **Deployment**: Configured for Render/Heroku (Procfile and render.yaml included)

## 📋 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the platform**:
   Open your browser and navigate to `http://localhost:5002`.

## 📂 Project Structure

- `app.py`: Main Flask application and API endpoints.
- `mltest.py`: Machine learning training and evaluation script.
- `static/`: Frontend assets (CSS, JS, images).
- `templates/`: HTML templates.
- `*.pkl`: Serialized ML models and encoders.
- `requirements.txt`: Project dependencies.

## 💡 Implementation Details

The system uses a **Random Forest Classifier** trained on historical agricultural data. Categorical inputs (Crop, State, Season) are transformed using a One-Hot Encoder, ensuring the model's accuracy across diverse agricultural contexts.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📄 Research paper: leakage-aware benchmark

The `paper_experiments/` folder contains the code and results for our paper
*"How Accurate Is Crop-Yield Prediction Really? A Leakage-Aware Benchmark on Indian State- and District-Level Data"*.

```bash
cd paper_experiments
pip install -r requirements.txt   # macOS: also run `brew install libomp` for XGBoost
python run_all.py                 # state-level study, ~5 minutes
python run_district.py            # district-level study, ~12 minutes (download the data first, see paper_experiments/README.md)
```

| Experiment | Output |
|---|---|
| E0 Data audit (fertilizer/pesticide = area × yearly rate) | `results/tables/E0_*` |
| E1 Leakage cascade (95.7% → 80.7%) | `results/tables/E1_*`, `results/figures/fig1_*` |
| E2 Regression benchmark, time split, Wilcoxon tests | `results/tables/E2_*`, `results/figures/fig3_*` |
| E3 Feature ablation + SHAP | `results/tables/E3_*`, `results/figures/fig2_*` |
| E4 Unseen-state generalisation | `results/tables/E4_*` |
| D0–D4 District-level replication (official APY data) | `results/district/tables/D*` |

Seeds 0–4; train 1997–2014, test 2015–2020; cluster bootstrap (1000 resamples) and series-level Wilcoxon tests. Package versions are pinned in `paper_experiments/requirements.txt`.

> **Note on the web demo:** the Flask app uses the original model, which takes *Production* as an input.
> Because Yield = Production ÷ Area, this feature leaks the answer, so the demo's accuracy (~95%) is
> optimistic. See the paper for an honest evaluation (80.7% at state level and 77.4% at district level after removing the shortcuts).

**Dataset:** [Crop Yield in Indian States (Kaggle)](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset)
