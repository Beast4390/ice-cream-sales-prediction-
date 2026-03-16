# 🍦 Ice Cream Sales Prediction Dashboard (SaaS Edition)


 **The App Demo link**- https://dbpr7bizawsehabv7t9agz.streamlit.app/


## 🚀 Product Summary
**Ice Cream Sales Prediction Dashboard** is a SaaS-ready analytics product designed for ice cream shop owners and operations teams. It turns historical temperature and sales data into actionable business insights, forecast scenarios, and downloadable reports.

**Built on:** Python, scikit-learn, Streamlit, Plotly.

---

## 🧠 Business Value
- **Forecast revenue** using temperature-driven models
- **Optimize inventory** using demand-level insights (high / medium / low)
- **Run what-if exercises** with temperature, pricing, and cost inputs
- **Download shareable reports** for executive decision-making
- **Support custom datasets** via upload + automatic model retraining

---

## ✨ Dashboard Highlights
### Premium Analytics Layout
- Top analytics header + KPI cards
- Clear section separation (forecast, charts, business insights)
- Consistent dark theme and modern styling

### Advanced Prediction Panel
- Temperature slider, selling price input, cost input
- Revenue prediction (INR), unit sales, profit (INR)
- Business recommendation engine with demand tiers

### Scenario Simulation
- Automatic “What-If” table for temperatures: 20, 25, 30, 35, 40°C
- Includes revenue, units, profit, demand level
- Downloadable forecast report

### Forecast Analytics (Interactive Charts)
- Revenue vs Temperature forecast
- Regression fit & residual error analysis
- Revenue distribution histogram
- Temperature demand segmentation

### Model Performance Insights
- R² score and Mean Squared Error
- Training sample size
- Peak sales temperature insight

### Data & Reporting
- Upload custom CSV datasets
- Automatic retraining + updated insights
- Download full report (HTML) with charts and tables

---

## 🗂️ Project Structure
```
Ice-cream-sales/
│
├── app/
│   └── streamlit_app.py          # Main SaaS dashboard app
│
├── data/
│   └── IceCreamData.csv          # Default dataset
│
├── models/
│   └── icecream_model.pkl        # Trained model artifact
│
├── notebooks/
│   └── analysis.ipynb            # Exploratory analysis
│
├── outputs/
│   ├── scatter_plot.png          # Legacy training plots
│   └── regression_plot.png       # Legacy training plots
│
├── src/
│   ├── data_preprocessing.py     # Data loading & cleaning
│   ├── train_model.py            # Training + retraining logic
│   ├── predict.py                # CLI prediction tool
│   └── utils.py                  # Helper utilities
│
├── requirements.txt               # Dependencies
└── README.md                      # Product documentation
```

---

## 🧰 Installation & Setup
### Requirements
- Python 3.8+
- Virtual environment (recommended)

### Install
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Run the Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 📥 Uploading Your Own Dataset
1. Click **Upload dataset (CSV)** in the sidebar.
2. Upload a file containing at least **Temperature** and **Revenue** columns.
3. The app will retrain the model and refresh insights automatically.

---

## 📄 Report Generation
Click **Download Sales Forecast Report** to export an interactive HTML report containing:
- KPI summary
- Scenario simulation table
- Charts and model performance metrics

---

## 🔮 Future SaaS Extensions (Roadmap)
- 🌦️ Weather API integration for real-time forecasting
- 🏬 Multi-store analytics + regional comparisons
- 📈 Multi-factor demand modeling (promotions, holidays, foot traffic)
- 📩 Automated daily/weekly email reports
- ☁️ Deployment to cloud platforms (Streamlit Cloud, AWS, Azure)

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes & test
4. Submit a pull request

---

**Built as a lightweight, polished analytics product using Streamlit and Plotly.**
