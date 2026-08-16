"""Tests for the split, baseline, metrics and pipeline-construction logic.

Model quality itself (RMSE 4.79 etc.) is only meaningful on the real 3M row gold
table and is reported in reports/model_metrics.log from an actual run, not
reproduced here. What these tests pin down is the plumbing the README's claims
depend on: that the split cannot leak future rows into training, that the
baseline's fallback behaves the way it is described, and that the metrics
returned aren't silently mislabeled.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import Row

from src.train_pipeline import (
    build_baseline,
    build_ml_pipeline,
    evaluate_predictions,
    time_based_split,
)


def ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


class TestTimeBasedSplit:
    @pytest.fixture
    def ten_rows(self, spark):
        rows = [
            Row(pickup_ts=ts(f"2023-01-{day:02d} 09:00:00"), fare_amount=float(day))
            for day in range(1, 11)
        ]
        return spark.createDataFrame(rows)

    def test_every_row_lands_in_exactly_one_side(self, ten_rows):
        train_df, test_df = time_based_split(ten_rows, train_ratio=0.8)
        assert train_df.count() + test_df.count() == ten_rows.count()

    def test_no_test_row_precedes_a_train_row(self, ten_rows):
        """Pins the property the README's leakage claim rests on.

        A future refactor that switches this to a random sample instead of a
        time-ordered one would still pass a naive row-count check, but this
        assertion would catch it immediately.
        """
        train_df, test_df = time_based_split(ten_rows, train_ratio=0.8)
        max_train_ts = train_df.agg({"pickup_ts": "max"}).first()[0]
        min_test_ts = test_df.agg({"pickup_ts": "min"}).first()[0]
        assert max_train_ts < min_test_ts

    def test_split_respects_the_requested_ratio(self, spark):
        rows = [
            Row(pickup_ts=ts("2023-01-01 00:00:00") + dt.timedelta(hours=i), fare_amount=float(i))
            for i in range(100)
        ]
        df = spark.createDataFrame(rows)
        train_df, test_df = time_based_split(df, train_ratio=0.8)
        train_share = train_df.count() / df.count()
        assert 0.75 <= train_share <= 0.85


class TestBuildBaseline:
    @pytest.fixture
    def train_df(self, spark):
        return spark.createDataFrame(
            [
                Row(pickup_zone="A", pickup_ts=ts("2023-01-01 09:00:00"), fare_amount=10.0),
                Row(pickup_zone="A", pickup_ts=ts("2023-01-01 09:15:00"), fare_amount=20.0),
                Row(pickup_zone="B", pickup_ts=ts("2023-01-01 14:00:00"), fare_amount=50.0),
            ]
        )

    def test_a_seen_zone_hour_gets_its_own_group_mean(self, spark, train_df):
        test_df = spark.createDataFrame(
            [Row(pickup_zone="A", pickup_ts=ts("2023-01-02 09:05:00"), fare_amount=999.0)]
        )
        result = build_baseline(train_df, test_df).collect()[0]
        assert result["prediction"] == pytest.approx(15.0)

    def test_an_unseen_zone_hour_falls_back_to_the_global_mean(self, spark, train_df):
        """train fares are 10, 20, 50 so the global mean is 80/3."""
        test_df = spark.createDataFrame(
            [Row(pickup_zone="C", pickup_ts=ts("2023-01-02 03:00:00"), fare_amount=999.0)]
        )
        result = build_baseline(train_df, test_df).collect()[0]
        assert result["prediction"] == pytest.approx(80.0 / 3.0)

    def test_prediction_ignores_the_test_rows_own_fare(self, spark, train_df):
        """The baseline model is fit on train_df only.

        If it were accidentally computed from train plus test (or from test
        alone), changing the test row's fare would change its own prediction.
        It must not.
        """
        low = spark.createDataFrame(
            [Row(pickup_zone="A", pickup_ts=ts("2023-01-02 09:05:00"), fare_amount=1.0)]
        )
        high = spark.createDataFrame(
            [Row(pickup_zone="A", pickup_ts=ts("2023-01-02 09:05:00"), fare_amount=9999.0)]
        )
        pred_low = build_baseline(train_df, low).collect()[0]["prediction"]
        pred_high = build_baseline(train_df, high).collect()[0]["prediction"]
        assert pred_low == pytest.approx(pred_high)
        assert pred_low == pytest.approx(15.0)


class TestEvaluatePredictions:
    def test_metrics_match_hand_computed_values(self, spark):
        # actual: 10, 20, 30 / predicted: 12, 18, 33 -> errors -2, 2, -3
        rows = [
            Row(fare_amount=10.0, prediction=12.0),
            Row(fare_amount=20.0, prediction=18.0),
            Row(fare_amount=30.0, prediction=33.0),
        ]
        df = spark.createDataFrame(rows)
        metrics = evaluate_predictions(df)

        errors = [-2.0, 2.0, -3.0]
        expected_mae = sum(abs(e) for e in errors) / len(errors)
        expected_rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        mean_actual = sum(r["fare_amount"] for r in rows) / len(rows)
        ss_res = sum(e**2 for e in errors)
        ss_tot = sum((r["fare_amount"] - mean_actual) ** 2 for r in rows)
        expected_r2 = 1 - ss_res / ss_tot

        assert metrics["MAE"] == pytest.approx(expected_mae, abs=1e-6)
        assert metrics["RMSE"] == pytest.approx(expected_rmse, abs=1e-6)
        assert metrics["R2"] == pytest.approx(expected_r2, abs=1e-6)

    def test_keys_are_exactly_rmse_mae_r2(self, spark):
        df = spark.createDataFrame([Row(fare_amount=1.0, prediction=1.0)])
        assert set(evaluate_predictions(df).keys()) == {"RMSE", "MAE", "R2"}


class TestBuildMlPipeline:
    def test_stages_are_the_expected_types_and_order(self):
        pipeline, rf = build_ml_pipeline()
        stage_types = [type(stage).__name__ for stage in pipeline.getStages()]
        assert stage_types == [
            "SQLTransformer",
            "StringIndexer",
            "OneHotEncoder",
            "FeatureHasher",
            "VectorAssembler",
            "RandomForestRegressor",
        ]
        assert pipeline.getStages()[-1] is rf

    def test_pipeline_fits_and_transforms_a_toy_frame_end_to_end(self, spark):
        """Exercises the SQLTransformer -> StringIndexer -> OHE -> FeatureHasher ->
        VectorAssembler chain on real (tiny) data, since that interaction is exactly
        where a column name or handleInvalid mismatch would silently break at scale.
        """
        rows = [
            Row(
                pickup_ts=ts("2023-01-01 09:00:00"),
                pickup_borough="Manhattan",
                pickup_zone="Midtown",
                dropoff_borough="Queens",
                dropoff_zone="JFK",
                trip_distance_miles=3.0,
                passenger_count=1,
                trip_seq_in_zone_day=1,
                time_since_prev_pickup_minutes=0.0,
                fare_amount=15.0,
            ),
            Row(
                pickup_ts=ts("2023-01-01 18:00:00"),
                pickup_borough="Brooklyn",
                pickup_zone="Park Slope",
                dropoff_borough="Manhattan",
                dropoff_zone="Midtown",
                trip_distance_miles=5.5,
                passenger_count=2,
                trip_seq_in_zone_day=1,
                time_since_prev_pickup_minutes=12.0,
                fare_amount=22.0,
            ),
            Row(
                pickup_ts=ts("2023-01-02 07:30:00"),
                pickup_borough="Queens",
                pickup_zone="JFK",
                dropoff_borough="Manhattan",
                dropoff_zone="Midtown",
                trip_distance_miles=14.0,
                passenger_count=1,
                trip_seq_in_zone_day=2,
                time_since_prev_pickup_minutes=45.0,
                fare_amount=52.0,
            ),
        ]
        df = spark.createDataFrame(rows)
        pipeline, _ = build_ml_pipeline()
        model = pipeline.fit(df)
        predictions = model.transform(df)

        assert "prediction" in predictions.columns
        assert predictions.count() == 3
        assert all(p is not None for p in [r["prediction"] for r in predictions.collect()])
