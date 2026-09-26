# 🌾 AI-Powered Crop Yield Prediction and Optimization

An end-to-end **Machine Learning-based Crop Yield Prediction and Optimization System** that predicts crop yield in tons per hectare using soil, weather, remote-sensing, and farm-management parameters.

The project compares multiple regression algorithms, selects the best-performing model, provides an interactive Streamlit dashboard for prediction, and recommends optimized combinations of controllable agricultural inputs such as fertilizer, irrigation, pesticide, and planting density.

---

## 📌 Project Overview

Agricultural yield depends on several factors such as:

* Crop type
* Region
* Soil type
* Rainfall
* Temperature
* Humidity
* Sunlight
* Soil pH
* Nitrogen, Phosphorus and Potassium
* Organic matter
* Fertilizer usage
* Irrigation
* Pesticide usage
* NDVI (Normalized Difference Vegetation Index)
* Planting density
* Days to harvest

This project uses these parameters to train regression models that estimate crop yield in **tons/hectare**.

In addition to prediction, the system uses the trained ML model as a simulator and performs a grid-search optimization over controllable inputs to identify combinations that can maximize predicted yield, optionally subject to a per-hectare budget.

---

## ✨ Key Features

### 🤖 Machine Learning Prediction

Predicts crop yield using agricultural, weather, soil, remote-sensing, and management parameters.

### 📊 Multiple Model Comparison

The project trains and compares:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

The best model is selected according to the **R² score** on the test set.

### 🔄 Data Preprocessing

The ML pipeline uses:

* `StandardScaler` for numerical features
* `OneHotEncoder` for categorical features
* `ColumnTransformer`
* Scikit-learn `Pipeline`

### 📈 Model Evaluation

The models are evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

A **5-fold cross-validation** is also performed on the selected model to check model robustness.

### 🎯 Input Optimization

The optimization module searches combinations of:

* Fertilizer
* Irrigation
* Pesticide
* Planting density

It ranks combinations according to predicted yield.

An optional budget constraint can also be applied.

### 📉 Sensitivity Analysis

The application generates yield-response curves showing how predicted yield changes when individual controllable inputs are varied.

### 🖥️ Interactive Streamlit Dashboard

The dashboard allows users to enter field conditions and immediately view:

* Predicted crop yield
* Optimized input combinations
* Estimated input cost
* Potential yield improvement
* Sensitivity analysis charts

### 💻 Command-Line Prediction

The project also provides a command-line prediction script for making predictions without opening the dashboard.

---

## 🧠 Machine Learning Workflow

```text
Agricultural Dataset
        ↓
Data Loading
        ↓
Feature Selection
        ↓
Numerical + Categorical Preprocessing
        ↓
StandardScaler + OneHotEncoder
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Linear Regression
Random Forest
Gradient Boosting
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
5-Fold Cross Validation
        ↓
Save Trained Pipeline
        ↓
Prediction / Optimization
        ↓
Streamlit Dashboard
```

---

## 📊 Input Features

### Categorical Features

| Feature     | Description         |
| ----------- | ------------------- |
| `crop`      | Crop type           |
| `region`    | Geographical region |
| `soil_type` | Soil type           |

### Weather Features

| Feature          | Unit      |
| ---------------- | --------- |
| `rainfall_mm`    | mm        |
| `avg_temp_c`     | °C        |
| `humidity_pct`   | %         |
| `sunlight_hours` | hours/day |

### Soil Features

| Feature                   | Unit  |
| ------------------------- | ----- |
| `soil_ph`                 | pH    |
| `soil_nitrogen_kg_ha`     | kg/ha |
| `soil_phosphorus_kg_ha`   | kg/ha |
| `soil_potassium_kg_ha`    | kg/ha |
| `soil_organic_matter_pct` | %     |

### Farm Management Features

| Feature                     | Unit               |
| --------------------------- | ------------------ |
| `fertilizer_kg_ha`          | kg/ha              |
| `irrigation_mm`             | mm                 |
| `pesticide_l_ha`            | L/ha               |
| `planting_density_k_per_ha` | thousand plants/ha |
| `days_to_harvest`           | days               |

### Remote Sensing

| Feature | Description                            |
| ------- | -------------------------------------- |
| `ndvi`  | Normalized Difference Vegetation Index |

### Target Variable

```text
yield_tons_per_ha
```

The training script uses the above categorical and numerical features and predicts `yield_tons_per_ha`.

---

## 🧪 Machine Learning Models

### 1. Linear Regression

Used as a baseline regression model.

### 2. Random Forest Regressor

An ensemble learning model using multiple decision trees.

Configuration includes:

```text
n_estimators = 300
max_depth = 12
min_samples_leaf = 3
```

### 3. Gradient Boosting Regressor

An ensemble boosting model used as another candidate regression model.

Configuration includes:

```text
n_estimators = 300
max_depth = 3
learning_rate = 0.05
```

The project selects the best-performing model according to test-set R² and then performs 5-fold cross-validation.

---

## 🔧 Optimization Module

The optimization module treats the trained ML model as a simulator.

The following inputs are treated as controllable:

```text
Fertilizer
Irrigation
Pesticide
Planting Density
```

The module performs a grid search over predefined ranges and calculates predicted yield for each combination.

It can also calculate an approximate input cost and filter combinations according to an optional budget.

Example:

```text
Fixed conditions
      ↓
Generate input combinations
      ↓
Predict yield for every combination
      ↓
Calculate input cost
      ↓
Apply optional budget constraint
      ↓
Sort by predicted yield
      ↓
Return top combinations
```

The current implementation returns the top combinations ranked by predicted yield.

---

## 📈 Sensitivity Analysis

Sensitivity analysis keeps other conditions fixed and changes one controllable variable at a time.

The system generates yield-response curves for:

* Fertilizer
* Irrigation
* Pesticide
* Planting density

This helps visualize how the trained model responds to changes in individual agricultural inputs.

---

## 🖥️ Streamlit Dashboard

The application provides controls for:

### Field Conditions

* Crop
* Region
* Soil type

### Weather

* Rainfall
* Average temperature
* Humidity
* Sunlight hours

### Soil

* Soil pH
* Nitrogen
* Phosphorus
* Potassium
* Organic matter

### Remote Sensing

* NDVI
* Days to harvest

### Farm Management

* Fertilizer
* Irrigation
* Pesticide
* Planting density

### Optimization

* Optional budget per hectare

The dashboard then displays the predicted yield, optimized input combinations, estimated cost, and sensitivity charts.

---

## 📁 Project Structure

Recommended working structure:

```text
Crop_Yield_Prediction_Using_AI/
│
├── app_streamlit.py
├── train_model.py
├── predict.py
├── optimize.py
├── generate_data.py
├── requirements.txt
├── run_dashboard.bat
│
├── data/
│   └── crop_yield_data.csv
│
├── models/
│   ├── crop_yield_model.pkl
│   ├── metrics.json
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
│
└── README.md
```

The repository currently contains the main Python scripts, dataset/model files, generated plots, requirements file, and Windows dashboard launcher. The training code creates the `models` directory and writes the trained pipeline and evaluation artifacts there.

---

# 🚀 Installation and Execution

## Step 1 — Clone the Repository

Open Git Bash or Command Prompt:

```bash
git clone https://github.com/soumyahallur02-cpu/Crop_Yield_Prediction_Using_AI.git
```

Move into the project:

```bash
cd Crop_Yield_Prediction_Using_AI
```

---

## Step 2 — Check Python

Run:

```bash
python --version
```

Recommended:

```text
Python 3.10+
```

Then verify pip:

```bash
pip --version
```

---

## Step 3 — Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

---

## Step 4 — Activate the Virtual Environment

### Windows Command Prompt

```cmd
venv\Scripts\activate
```

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Git Bash

```bash
source venv/Scripts/activate
```

After activation, you should see:

```text
(venv)
```

at the beginning of your terminal line.

---

## Step 5 — Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

The repository's requirements currently specify NumPy, Pandas, Scikit-learn, Matplotlib, Joblib, and Streamlit.

If necessary, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

Then:

```bash
pip install -r requirements.txt
```

---

# 📂 Step 6 — Prepare the Data Folder

Create:

```text
data
```

inside the project directory.

Your final path should be:

```text
Crop_Yield_Prediction_Using_AI/data/
```

The training script expects:

```text
data/crop_yield_data.csv
```

The repository already contains `generate_data.py`, which generates a synthetic agricultural dataset containing 6,000 records.

---

# 🌱 Step 7 — Generate the Dataset

If you want to generate the dataset yourself, run:

```bash
python generate_data.py
```

### Important

The current `generate_data.py` contains a hard-coded output path:

```text
/home/claude/crop_yield_project/data/crop_yield_data.csv
```

That path is not appropriate for a normal Windows local setup.

Therefore, for Windows, change the final part of `generate_data.py` to:

```python
import os

os.makedirs("data", exist_ok=True)

df = generate_dataset()

out_path = "data/crop_yield_data.csv"

df.to_csv(out_path, index=False)

print(f"Generated {len(df)} rows -> {out_path}")
```

Then run:

```bash
python generate_data.py
```

You should get:

```text
data/crop_yield_data.csv
```

---

# 🤖 Step 8 — Train the Machine Learning Models

Run:

```bash
python train_model.py
```

The training script:

1. Loads the dataset.
2. Selects numerical and categorical features.
3. Splits the data into 80% training and 20% testing.
4. Applies preprocessing.
5. Trains Linear Regression.
6. Trains Random Forest.
7. Trains Gradient Boosting.
8. Calculates MAE, RMSE and R².
9. Selects the best model based on R².
10. Performs 5-fold cross-validation.
11. Saves the trained model.
12. Generates evaluation plots.

This workflow is implemented directly in your `train_model.py`.

After successful training, you should have:

```text
models/
│
├── crop_yield_model.pkl
├── metrics.json
├── model_comparison.png
├── feature_importance.png
└── predicted_vs_actual.png
```

---

# 🔮 Step 9 — Test Prediction from Command Line

After training, you can use:

```bash
python predict.py
```

However, `predict.py` requires all input parameters as command-line arguments.

Example:

```bash
python predict.py ^
--crop Maize ^
--region Central ^
--soil_type Loamy ^
--rainfall_mm 550 ^
--avg_temp_c 25 ^
--humidity_pct 60 ^
--sunlight_hours 7.5 ^
--soil_ph 6.4 ^
--soil_nitrogen_kg_ha 55 ^
--soil_phosphorus_kg_ha 38 ^
--soil_potassium_kg_ha 42 ^
--soil_organic_matter_pct 3.2 ^
--fertilizer_kg_ha 150 ^
--irrigation_mm 300 ^
--pesticide_l_ha 2 ^
--ndvi 0.65 ^
--planting_density_k_per_ha 85 ^
--days_to_harvest 120
```

The script loads the trained model, predicts yield, and then displays the top optimized input combinations.

---

# 🎯 Step 10 — Run the Optimization Module

You can also run:

```bash
python optimize.py
```

The script includes an example field and performs optimization both without a budget and with a `$250/ha` budget example.

The optimization searches combinations of:

```text
Fertilizer
Irrigation
Pesticide
Planting density
```

and returns the combinations with the highest predicted yield.

---

# 🖥️ Step 11 — Run the Streamlit Dashboard

This is the main application.

Run:

```bash
streamlit run app_streamlit.py
```

Streamlit will display a local URL, usually similar to:

```text
http://localhost:8501
```

Open that address in your browser.

Your repository also contains `run_dashboard.bat`, which runs:

```text
streamlit run app_streamlit.py
```

so on Windows you can double-click the batch file after the environment and dependencies are correctly configured.

---

# 🌾 Step 12 — Use the Dashboard

After opening the dashboard:

### 1. Select Crop

Choose:

```text
Wheat
Rice
Maize
Soybean
Cotton
```

### 2. Select Region

Choose:

```text
North
South
East
West
Central
```

### 3. Select Soil Type

Choose:

```text
Loamy
Sandy
Clayey
Silty
```

### 4. Enter Weather Conditions

Adjust:

* Rainfall
* Temperature
* Humidity
* Sunlight

### 5. Enter Soil Conditions

Adjust:

* pH
* Nitrogen
* Phosphorus
* Potassium
* Organic matter

### 6. Enter Remote-Sensing Information

Adjust:

* NDVI
* Days to harvest

### 7. Enter Management Inputs

Adjust:

* Fertilizer
* Irrigation
* Pesticide
* Planting density

### 8. Optional Budget

Enter an optimization budget in:

```text
$/ha
```

Set it to `0` if you do not want a budget constraint.

### 9. View Prediction

The dashboard displays:

```text
Predicted Yield
```

in:

```text
tons/hectare
```

### 10. View Optimization

The system displays the top input combinations and their predicted yields and estimated input costs.

### 11. View Sensitivity

The dashboard displays charts showing how yield changes as individual controllable inputs are varied.

---

# 🧩 Technologies Used

| Category            | Technologies                                        |
| ------------------- | --------------------------------------------------- |
| Programming         | Python                                              |
| Data Processing     | Pandas, NumPy                                       |
| Machine Learning    | Scikit-learn                                        |
| Models              | Linear Regression, Random Forest, Gradient Boosting |
| Preprocessing       | StandardScaler, OneHotEncoder, ColumnTransformer    |
| Evaluation          | MAE, RMSE, R², 5-Fold Cross Validation              |
| Visualization       | Matplotlib, Streamlit charts                        |
| Deployment/UI       | Streamlit                                           |
| Model Serialization | Joblib                                              |
| Optimization        | Grid Search                                         |
| Version Control     | Git, GitHub                                         |

---

# 📊 Model Evaluation

The project evaluates each model using:

### MAE

Measures the average absolute prediction error.

### RMSE

Measures the square-rooted average squared prediction error.

### R²

Measures how well the model explains the variation in crop yield.

The best model is automatically selected using the highest test-set R² score. The selected pipeline is then evaluated using 5-fold cross-validation.

---

# 📌 Important Note About the Dataset

The current `generate_data.py` describes the dataset as **synthetic data** generated to model realistic agronomic relationships. It uses crop-specific base yields and ideal rainfall/temperature values and adds controlled variation and noise.

For a production deployment, the project can be adapted to use real:

* Soil sensor data
* Weather API data
* Farm records
* Satellite-derived NDVI
* Historical crop yield records

---

# 🔮 Future Enhancements

* Integrate real-time weather APIs such as OpenWeather or NASA POWER.
* Integrate satellite imagery and automatically calculate NDVI.
* Add real farm datasets.
* Add crop recommendation.
* Add disease and pest-risk prediction.
* Add weather forecasting.
* Add database support for storing prediction history.
* Add user authentication.
* Deploy the Streamlit application to a cloud platform.
* Replace grid search with advanced optimization techniques such as Bayesian Optimization.
* Add explainable AI techniques such as SHAP.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes. Predictions are generated by a machine-learning model and should not be treated as guaranteed agricultural outcomes.

For real-world deployment, the model should be retrained and validated using reliable local farm, soil, weather, and crop-yield data.

---

## 👩‍💻 Author

**Soumya Hallur**

Computer Science Engineering Student

GitHub:
https://github.com/soumyahallur02-cpu

---

## ⭐ Project Repository

https://github.com/soumyahallur02-cpu/Crop_Yield_Prediction_Using_AI
