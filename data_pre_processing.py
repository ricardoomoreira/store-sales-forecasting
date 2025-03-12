import pickle

import numpy as np
import pandas as pd
import torch


def create_features_dataset(
    df, df_stores, df_oil, nat_holidays, reg_holidays, loc_holidays
):
    df_out = df.copy()
    # Get the number of days passed since the previous payday (15th and end of month).
    df_out["dist_to_payday"] = pd.to_datetime(df["date"]).dt.day.apply(
        lambda x: x if x < 15 else x - 15
    )
    df_out["weekday"] = pd.to_datetime(df["date"]).dt.dayofweek
    df_out["month"] = pd.to_datetime(df["date"]).dt.month - 1  # 0-indexed
    df_out["oil_price"] = (
        df.set_index("date").join(df_oil.set_index("date"))["dcoilwtico"].values
    )
    stores_cols = df_stores.set_index("store_nbr").columns
    df_out[stores_cols] = (
        df.set_index("store_nbr")
        .join(df_stores.set_index("store_nbr"))[stores_cols]
        .values
    )
    df_out["nat_holiday"] = (
        df_out.set_index("date")
        .join(nat_holidays.drop_duplicates("date").set_index("date"), rsuffix="_holi")[
            "type_holi"
        ]
        .notna()
        .values
    )
    df_out["reg_holiday"] = (
        df_out.set_index(["date", "state"])
        .join(
            reg_holidays.drop_duplicates(["date", "state"]).set_index(
                ["date", "state"]
            ),
            rsuffix="_holi",
        )["type_holi"]
        .notna()
        .values
    )
    df_out["loc_holiday"] = (
        df_out.set_index(["date", "city"])
        .join(
            loc_holidays.drop_duplicates(["date", "city"]).set_index(["date", "city"]),
            rsuffix="_holi",
        )["type_holi"]
        .notna()
        .values
    )
    df_out["is_holiday"] = (
        df_out["nat_holiday"] | df_out["reg_holiday"] | df_out["loc_holiday"]
    )

    return df_out


def main():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_oil = pd.read_csv("oil.csv")
    df_stores = pd.read_csv("stores.csv")
    df_holidays = pd.read_csv("holidays_events.csv")

    # Remove transferred holidays (check: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
    df_holidays = df_holidays[~df_holidays["transferred"]]

    # Breakdown holidays by locale.
    nat_holidays = df_holidays[df_holidays["locale"] == "National"]
    reg_holidays = df_holidays[df_holidays["locale"] == "Regional"].rename(
        columns={"locale_name": "state"}
    )
    loc_holidays = df_holidays[df_holidays["locale"] == "Local"].rename(
        columns={"locale_name": "city"}
    )

    # Check how many holidays are in test set.
    print(
        df_holidays[
            (pd.to_datetime(df_holidays["date"]) < "2017-08-31")
            & ("2017-08-15" < pd.to_datetime(df_holidays["date"]))
        ]
    )

    # REMARK: There is only 1 holiday is in the test set, to maximize performance,
    # we'll not try to predict holiday patterns, we'll just use the information of how
    # far or how close we are to the next holiday.

    df_tr = create_features_dataset(
        df_train, df_stores, df_oil, nat_holidays, reg_holidays, loc_holidays
    )
    df_te = create_features_dataset(
        df_test, df_stores, df_oil, nat_holidays, reg_holidays, loc_holidays
    )

    df_merged = pd.concat([df_tr, df_te]).reset_index(drop=True)

    # Save merged dataset to parquet.
    df_merged.to_parquet("merged.parquet", index=False)

    STORE_TO_I = {s: i for i, s in enumerate(df_tr["store_nbr"].unique())}
    I_TO_STORE = {i: s for i, s in enumerate(df_tr["store_nbr"].unique())}
    FAM_TO_I = {f: i for i, f in enumerate(df_tr["family"].unique())}
    I_TO_FAM = {i: f for i, f in enumerate(df_tr["family"].unique())}
    PROMO_TO_I = {True: 1, False: 0}
    I_TO_PROMO = {1: True, 0: False}
    MONTH_TO_I = {m: i for i, m in enumerate(df_tr["month"].unique())}
    I_TO_MONTH = {i: m for i, m in enumerate(df_tr["month"].unique())}
    CITY_TO_I = {c: i for i, c in enumerate(df_tr["city"].unique())}
    I_TO_CITY = {i: c for i, c in enumerate(df_tr["city"].unique())}
    STATE_TO_I = {s: i for i, s in enumerate(df_tr["state"].unique())}
    I_TO_STATE = {i: s for i, s in enumerate(df_tr["state"].unique())}
    TYPE_TO_I = {t: i for i, t in enumerate(df_tr["type"].unique())}
    I_TO_TYPE = {i: t for i, t in enumerate(df_tr["type"].unique())}
    CLUSTER_TO_I = {c: i for i, c in enumerate(df_tr["cluster"].unique())}
    I_TO_CLUSTER = {i: c for i, c in enumerate(df_tr["cluster"].unique())}

    FORECAST_WINDOW = df_te["date"].nunique()

    COLS = [
        "store",
        "family",
        "sales",
        "is_promotion",
        "dist_to_payday",
        "dist_to_next_holiday",
        "weekday",
        "month",
        "oil_price",
        "city",
        "state",
        "type",
        "cluster",
    ]

    def build_sequences(
        df: pd.DataFrame,
        grp_cols: list,
        forecast_window: int = 16,
        val_start: str = "2017-07-31",
        test_start: str = "2017-08-16",
    ):
        seqs = []
        for i, (_, data) in enumerate(df.groupby(grp_cols)):
            print(i, end="\r")
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date")

            if i == 0:
                splits = np.where(
                    data["date"] < val_start,
                    "train",
                    np.where(
                        (data["date"] >= val_start) & (data["date"] < test_start),
                        "val",
                        "test",
                    ),
                )

            data["store"] = data["store_nbr"].map(STORE_TO_I)
            data["family"] = data["family"].map(FAM_TO_I)

            # Remove holidays by replacing with the previous week value.
            data["sales"] = np.where(
                data["is_holiday"], data["sales"].shift(7), data["sales"]
            )
            data["sales"] = data["sales"].fillna(0)
            data["is_promotion"] = (data["onpromotion"] > 0).map(PROMO_TO_I)

            # For each day, reindex the holiday dates to fill forward:
            data["next_holiday_date"] = data.loc[data["is_holiday"], "date"].reindex(
                data.index, method="bfill"
            )

            # Calculate the distance (in days) to the next holiday:
            data["dist_to_next_holiday"] = (
                data["next_holiday_date"] - data["date"]
            ).dt.days
            data["dist_to_next_holiday"] = data["dist_to_next_holiday"].fillna(
                round(data["dist_to_next_holiday"].mean())
            )
            data["month"] = data["month"].map(I_TO_MONTH)

            # Fillna with the previous value.
            data["oil_price"] = data["oil_price"].bfill()
            data["city"] = data["city"].map(CITY_TO_I)
            data["state"] = data["state"].map(STATE_TO_I)
            data["type"] = data["type"].map(TYPE_TO_I)
            data["cluster"] = data["cluster"].map(CLUSTER_TO_I)

            # New column with the sales of the upcomming n days.
            for i in range(1, forecast_window + 1):
                data[f"sales_next_{i}"] = data["sales"].shift(-i)
                data[f"sales_prev_{i}"] = data["sales"].shift(forecast_window - i)

            # Aggregate sales_next cols into a single list column.
            sales_prev = data[
                [f"sales_prev_{i}" for i in range(1, forecast_window + 1)]
            ].values
            sales_next = data[
                [f"sales_next_{i}" for i in range(1, forecast_window + 1)]
            ].values

            # Crop out the last forecast_window days since the sales_next are NaN.
            seq = (
                torch.tensor(data["id"].values),
                torch.tensor(data[COLS].values),
                torch.tensor(sales_prev),
                torch.tensor(sales_next),
            )
            seqs.append(seq)
        return seqs, splits

    seqs, splits = build_sequences(
        df_merged,
        ["store_nbr", "family"],
        forecast_window=FORECAST_WINDOW,
        val_start="2017-07-31",
        test_start="2017-08-16",
    )

    with open("seqs.pkl", "wb") as f:
        pickle.dump(
            (
                seqs,
                splits,
                FORECAST_WINDOW,
                COLS,
                STORE_TO_I,
                I_TO_STORE,
                FAM_TO_I,
                I_TO_FAM,
                PROMO_TO_I,
                I_TO_PROMO,
                MONTH_TO_I,
                I_TO_MONTH,
                CITY_TO_I,
                I_TO_CITY,
                STATE_TO_I,
                I_TO_STATE,
                TYPE_TO_I,
                I_TO_TYPE,
                CLUSTER_TO_I,
                I_TO_CLUSTER,
            ),
            f,
        )


if __name__ == "__main__":
    main()
