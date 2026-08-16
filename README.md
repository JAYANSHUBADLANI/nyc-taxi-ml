# NYC Taxi Fare Prediction at Scale (PySpark MLlib)

[![tests](https://github.com/JAYANSHUBADLANI/nyc-taxi-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/nyc-taxi-ml/actions/workflows/tests.yml)

I built this project as the machine learning companion to my PySpark data pipeline portfolio project. While the sibling project handles the data engineering (moving raw TLC taxi data through bronze, silver, and gold layers), this project handles the machine learning. 

The focus here is demonstrating distributed ML operations using Spark's Pipeline API, feature transformers, and model selection tools on a dataset that does not comfortably fit in a single machine's memory. I predict the `fare_amount` for taxi trips using context and geographical features from the gold table.

## Data Source

This model trains on a bounded one-month subset of real NYC TLC Yellow Taxi data (January 2023), producing approximately 3 million rows. The schema exactly matches the gold fact table output of my sibling pipeline project, complete with temporal window features and joined taxi zone lookups.

## The ML Pipeline

1. **Feature Engineering**: I use PySpark `SQLTransformer` to extract temporal features (hour, day of week, weekend flags) and hash the high-cardinality `zone_pair` column using `FeatureHasher` into a 1024-dimensional dense space to prevent Out-Of-Memory errors during tree splitting. Categorical zones are handled using `StringIndexer` and `OneHotEncoder`.
2. **Train/Test Split**: I split the dataset based on time (`pickup_ts`), holding out the last 20% of the timeline to prevent data leakage and simulate a real-world forecasting scenario. Random sampling would overstate accuracy by leaking information across correlated trips on the same day.
3. **Hyperparameter Tuning**: I use Spark's `TrainValidationSplit` over a small, deliberate grid (`maxDepth` in [5, 10], `numTrees` in [20]). Exhaustive grid searches at this scale are impractical, so a focused grid proves the mechanics of distributed tuning without wasting compute.
4. **Model**: A distributed `RandomForestRegressor`.

## Assessment & Results

Before quoting model metrics, I established a naive baseline predictor (the mean fare per pickup zone and hour bucket) computed the same way the model is evaluated. 

**Baseline Metrics:**
* RMSE: 12.10
* MAE: 7.37
* R²: 0.45

**Model Metrics (RandomForestRegressor):**
* RMSE: 4.79
* MAE: 2.17
* R²: 0.91

The model achieves a massive lift over the baseline, halving the RMSE and boosting R² from 0.45 to 0.91. 

**Error Breakdown & Weak Spots:**
Looking purely at the aggregate RMSE hides where the model fails. I broke down the absolute error by borough and hour of the day:
* **Geographical Weakness**: The highest errors occur for trips originating from "N/A" zones (avg absolute error of $43.39) and Newark Airport (EWR, avg error of $28.64). These likely represent flat-rate trips, out-of-bounds trips, or negotiated fares that the model cannot predict purely from duration/distance and time.
* **Temporal Weakness**: Errors spike between 4 AM and 6 AM, which coincides with the early morning shift changes and airport rush, causing unpredictable fare conditions compared to the rest of the day.

**Reproducibility:**
A model result isn't worth quoting if it depends on which run happened to finish. I refitted the final pipeline with the same parameters and random seed. The RMSE difference between the tuning run and the reproducibility run was exactly 0.000000. 

## Running the Pipeline

1. Verify you have a JDK installed (e.g., OpenJDK 17).
2. Set up the Python virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Generate the bounded gold data table:
   ```bash
   python src/data_generation.py
   ```
4. Run the full training pipeline (this takes several minutes on 3M rows):
   ```bash
   python src/train_pipeline.py --fraction 1.0 | tee reports/model_metrics.log
   ```

Every number quoted in the results section above comes from that run. The captured
output is committed at `reports/model_metrics.log` so the figures can be checked
against the run that produced them rather than taken on trust.
