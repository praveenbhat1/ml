# AgriTech Intelligence: Crop Yield Predictor

![AgriTech Intelligence Banner](https://img.shields.io/badge/AgriTech-Intelligence-1D9E75?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

AgriTech Intelligence is a high-precision machine learning classification system designed to predict crop yield potential. By analyzing environmental factors, cultivation data, and historical trends, the platform provides actionable insights for farmers and agricultural researchers.

## 🚀 Features

- **Real-time Classification**: Instant yield analysis (High/Low) using a pre-trained Random Forest model.
- **Data-Driven Insights**: Considers multiple variables such as Rainfall, Fertilizer, Pesticide usage, and Production area.
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
