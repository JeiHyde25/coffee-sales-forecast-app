
# ☕ Coffee Sales Forecast App

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-orange)

This is a Streamlit-based web application that predicts coffee sales based on historical CSV data using a linear regression model. It includes file upload support, future date forecasting, and performance metric output.

🚀 [Live Demo](https://coffee-forecast-app.onrender.com)

---

## 📁 Project Structure

```
coffee-sales-forecast-app/
├── data/                           # Input CSVs
├── src/
│   ├── notebook.ipynb              # Model development notebook
│   └── notebook.py                 # Main Streamlit app
├── Dockerfile                      # Container configuration
├── LICENSE                         # MIT License
├── README.md                       # This file
├── requirements.txt                # Python dependencies
```

---

## 🔧 Features

- 📈 Upload your coffee sales CSV data
- 📅 Predict future daily sales over N days
- 📊 View and download model predictions and metrics
- 🧪 View model performance (MAE, MSE, R²)

---

## 🚀 Tech Stack

- **Python 3.13**
- **Streamlit**
- **scikit-learn**
- **pandas**, **matplotlib**
- **Docker**
- **GitHub Actions** for CI/CD
- **Render** for deployment

---

## 🛠️ Getting Started (Local)

### 1. Clone the repository
```bash
git clone https://github.com/JeiHyde25/coffee-sales-forecast-app.git
cd coffee-sales-forecast-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run src/notebook.py
```

---

## 📦 Docker

To run in Docker:

```bash
docker build -t coffee-forecast-app .
docker run -p 8080:8080 coffee-forecast-app
```

---

## 📊 Model Performance

```
Mean Absolute Error : 175.22
Mean Squared Error  : 42268.66
R² Score             : 0.68
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**Harold Tago**  
[GitHub: JeiHyde25](https://github.com/JeiHyde25)

---

## 📫 Contact

For questions or collaborations, feel free to reach out via GitHub or [LinkedIn](https://www.linkedin.com/in/yourprofile)
