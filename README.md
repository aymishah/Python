# Car Price Prediction – End‑to‑End

This project trains a regression model to predict used car prices using the Kaggle dataset:
**Vehicle dataset from CarDekho** by *nehalbirla*.

> Dataset URL: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

## Project Structure

```
car_price_prediction_app/
├── data/
│   └── car_data.csv                # Put downloaded Kaggle CSV here (rename to car_data.csv)
├── models/
│   └── car_price_pipeline.pkl      # Will be created after training
├── reports/                        # Metrics & plots
├── src/
│   ├── train_regression.py         # Train + evaluate + save pipeline
│   └── utils.py                    # Helpers (plotting, binning)
├── app.py                          # Flask backend + HTML frontend
├── templates/
│   └── index.html                  # Simple web UI for predictions
├── static/
│   └── style.css                   # Minimal CSS
├── streamlit_app.py                # One-file GUI alternative
├── requirements.txt
└── README.md
```

## Quickstart

1. **Download the dataset CSV** from Kaggle and place it as:
   `./data/car_data.csv`

2. **Create a virtual environment** and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Train models** and generate plots/reports:
   ```bash
   python src/train_regression.py --data ./data/car_data.csv
   ```

   Outputs:
   - `models/car_price_pipeline.pkl` (sklearn Pipeline)
   - `reports/metrics.json`
   - `reports/residuals.png`, `reports/pred_vs_actual.png`, `reports/feature_importance.png`
   - `reports/confusion_matrix_bins.png` (classification view on binned prices)

4. **Run the Flask app** (backend + simple HTML frontend):
   ```bash
   python app.py
   # Open http://127.0.0.1:5000
   ```

5. **Or use Streamlit GUI**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Notes

- The dataset has columns like: `name`, `year`, `selling_price` (target), `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`. The training script infers available columns and handles missing values, encodes categoricals, scales numerics, and performs feature selection (Mutual Information).
- **Confusion matrix** is not defined for regression; we provide an *optional* binned-price classification view to visualize mistakes by price range.
- You can toggle models in the script: LinearRegression, RandomForestRegressor, GradientBoostingRegressor, and (optionally) XGBRegressor (requires `xgboost`).

## Repro Tips

- Use `--bins 5` (or another integer) to adjust price-range binning for the confusion matrix figure.
- Results will vary depending on preprocessing and random splits.
