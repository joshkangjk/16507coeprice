# Singapore COE Analytics & Predictor

An interactive web dashboard built with Streamlit that analyzes and forecasts Singapore Certificate of Entitlement (COE) vehicle premiums. 

This project was initially developed as part of a university exchange semester module and has been adapted into a full standalone web application. It uses a Random Forest machine learning model to predict price movements based on supply (quotas), consumer demand (bids), and macroeconomic inflation (CPI).

## Key Features
* **Scenario Modeling:** Interactive sliders allow users to adjust upcoming quotas, total expected bidders, and inflation metrics to see real-time price predictions.
* **Confidence Bands:** Utilizes ensemble tree variance to calculate 50% likely range boundaries for forecasted premiums.
* **Historical Trend Analysis:** Visualizes historical price volatility, driver correlations, and market tension (supply vs. demand pressure).
* **Automated Data Pipeline:** Pulls live historical COE bidding data and CPI metrics directly from the `data.gov.sg` API.

## Tech Stack
* **Language:** Python 3
* **Frontend/Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Requests
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`)
* **Data Visualization:** Plotly (`plotly.express`, `plotly.graph_objects`)

## How to Run Locally

To run this project on your local machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name