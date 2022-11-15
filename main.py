# Importing all dependencies
import pandas as pd
import numpy as np
from prophet import Prophet
from dataclasses import dataclass
import typing
from typing import Tuple
from flytekit.types.file import FlyteFile
from flytekit.types.file import PythonPickledFile


@dataclass
class Hyperparameters(object):
    """
    dataframe
    product
    end_perc
    """

    dataframe: str = "data.csv"
    product: str = "Product_1382"
    end_perc: float = 0.8


# Calling Hyperparameters class as an object
hp = Hyperparameters()

# Gathering data and creatin dataframe
def get_df(dataframe: str) -> pd.DataFrame:
    # Creating dataframe
    return pd.read_csv(dataframe)


def prepare_df(df: pd.DataFrame, product: str) -> pd.DataFrame:
    # Changing column values
    df["Date"] = pd.to_datetime(df["Date"])
    df["StateHoliday"] = df["StateHoliday"].replace(["a", "b"], "1")
    df["StateHoliday"] = df["StateHoliday"].astype(int)
    # Dropping the column Product_id
    df = df.drop(["Product_id"], axis=1)
    # Only considering one product
    df = df[df["Product_Code"] == hp.product]
    # Dropping the column Warehouse and Product_Code
    df = df.drop(["Product_Code", "Warehouse", "Product_Category"], axis=1)
    # Group by Date
    df = df.groupby("Date", as_index=False).agg(
        {
            "Order_Demand": "sum",
            "Open": "mean",
            "Promo": "mean",
            "StateHoliday": "mean",
            "SchoolHoliday": "mean",
            "Petrol_price": "mean",
        }
    )
    df.rename(columns={"Date": "ds", "Order_Demand": "y"}, inplace=True)
    return df


def split_df(df: pd.DataFrame, end_perc: float) -> pd.DataFrame:
    # Split dataset into train and test
    df_len = df.shape[0]
    end = int(df_len * end_perc)
    return (df.iloc[:end], df.iloc[end - 1 :])


### Declaring the exogenous variables to add to the model
exog_var = ["Open", "Promo", "StateHoliday", "SchoolHoliday", "Petrol_price"]


def create_model(train: pd.DataFrame) -> PythonPickledFile:
    # Building the model
    model = Prophet()
    for var in exog_var:
        model.add_regressor(var)
    # Fitting the model
    return model.fit(train)


def predict_future(model: PythonPickledFile, test: pd.DataFrame, df: pd.DataFrame) -> PythonPickledFile:
    # Creating future dates
    future = model.make_future_dataframe(periods=test.shape[0])
    future[exog_var] = df[exog_var]
    # Predicting future order values
    return model.predict(future)


def create_plot(
    forecast: pd.DataFrame, model: PythonPickledFile
) -> Tuple[FlyteFile[typing.TypeVar("png")], FlyteFile[typing.TypeVar("png")]]:
    return (
        model.plot(forecast).savefig("prediction.png"),
        model.plot_components(forecast).savefig("components.png"),
    )


def run_wf(
    dataframe: str, product: str, end_perc: float
) -> Tuple[FlyteFile[typing.TypeVar("png")], FlyteFile[typing.TypeVar("png")]]:
    """
    dataframe: hp.dataframe
    product: hp.product
    end_perc: hp.end_perc
    """
    train, test = split_df(prepare_df(get_df(dataframe), product), end_perc)
    return create_plot(predict_future(create_model(train), test, get_df(dataframe)),create_model(train))


if __name__ == "__main__":
    run_wf(hp.dataframe, hp.product, hp.end_perc)
