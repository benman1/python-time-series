"""Utility functions for data loading."""
import datetime
import operator
from typing import Literal, Sequence

import requests
import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastcache import lru_cache


SEASONALITY = Literal["hourly", "daily", "weekly", "monthly", "quarterly"]


def get_sample_data(
        seasonalities: Sequence[SEASONALITY],
        nsamples: int = 1000,
        use_trend: bool = True,
        composition: Literal["multiplicative", "additive"] = "additive",
) -> pd.DataFrame:
    """Create a sample dataset with seasonality at chosen periodicities.

    Based on https://www.statsmodels.org/dev/examples/notebooks/
    generated/mstl_decomposition.html
    """
    if len(seasonalities) == 0:
        return pd.DataFrame()
    op = operator.add if composition == "additive" else operator.mult
    freq_lookup = {
        "hourly": 1,
        "daily": 24,
        "weekly": 24 * 7,
        "monthly": 24 * 7 * 30,
        "quarterly": 24 * 7 * 30 * 3
    }

    t = np.arange(1, nsamples)
    # start with residual
    y = np.random.randn(len(t))
    if use_trend:
        y = op(y, 0.0001 * t**2)
    for seasonal in seasonalities:
        y = op(
            y,
            5 * np.sin(2 * np.pi * t / freq_lookup[seasonal])
        )
    ts = pd.date_range(start="2020-01-01", freq="H", periods=len(t))
    df = pd.DataFrame(data=y, index=ts, columns=["y"])
    return df


@lru_cache
def get_electrivity_demand() -> pd.DataFrame:
    """Get electricity demand in Victoria, Australia.

    From https://www.statsmodels.org/dev/examples/notebooks/
    generated/mstl_decomposition.html
    """
    url = (
        "https://raw.githubusercontent.com/tidyverts/"
        "tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
    )
    df = pd.read_csv(url)
    df["Date"] = df["Date"].apply(
        lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
    )
    df["ds"] = df["Date"] + pd.to_timedelta((df["Period"] - 1) * 30, unit="m")
    return df


@lru_cache(maxsize=1, typed=False)
def get_energy_demand(scale: bool = True) -> pd.DataFrame:
    resp = requests.get(
        "https://github.com/camroach87/gefcom2017data/"
        "raw/master/data/gefcom.rda",
        allow_redirects=True,
    )
    open("gefcom.rda", "wb").write(resp.content)
    result = pyreadr.read_r("gefcom.rda")
    df = result["gefcom"].pivot(index="ts", columns="zone", values="demand")
    df = df.asfreq("d")
    if not scale:
        return df
    return pd.DataFrame(
        data=StandardScaler().fit_transform(df),
        columns=df.columns, index=df.index
    )


@lru_cache
def get_pollution() -> pd.DataFrame:
    """Loads the OWID pollution dataset."""
    column = "Suspended Particulate Matter (SPM) (Fouquet and DPCC (2011))"
    df = pd.read_csv(
        "https://raw.githubusercontent.com/owid/owid-datasets/master/"
        "datasets/Air%20pollution%20by%20city%20-%20Fouquet"
        "%20and%20DPCC%20(2011)/Air%20pollution%20by%20city%20-%20"
        "Fouquet%20and%20DPCC%20(2011).csv"
    ).pivot_table(values=column, index="Year", columns="Entity")
    df = df.astype(float)
    df.index = pd.Series(df.index).apply(
        lambda x: datetime.strptime(str(x), "%Y")
    )
    return df


@lru_cache
def get_covid(column: str = "new_cases") -> pd.DataFrame:
    """Load COVID data from OWID.

    column can be one of these:
    'total_cases', 'new_cases',
    'new_cases_smoothed', 'total_deaths', 'new_deaths',
    'new_deaths_smoothed', 'total_cases_per_million',
    'new_cases_per_million', 'new_cases_smoothed_per_million',
    'total_deaths_per_million', 'new_deaths_per_million',
    'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients',
    'icu_patients_per_million', 'hosp_patients',
    'hosp_patients_per_million', 'weekly_icu_admissions',
    'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
    'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
    'total_tests_per_thousand', 'new_tests_per_thousand',
    'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
    'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations',
    'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
    'new_vaccinations', 'new_vaccinations_smoothed',
    'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
    'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
    'new_vaccinations_smoothed_per_million',
    'new_people_vaccinated_smoothed',
    'new_people_vaccinated_smoothed_per_hundred', 'stringency_index',
    'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
    'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate',
    'diabetes_prevalence', 'female_smokers', 'male_smokers',
    'handwashing_facilities', 'hospital_beds_per_thousand',
    'life_expectancy', 'human_development_index', 'population',
    'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
    'excess_mortality', 'excess_mortality_cumulative_per_million'
    """
    df = pd.read_csv(
        "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    ).pivot_table(values=column, index="date", columns="location")
    df["date"] = pd.to_datetime(df["date"])
    df.index = pd.to_datetime(df.index)
    df = df.fillna(0.0)
    return df


@lru_cache(maxsize=1, typed=False)
def get_ford(train: bool = True):
    """Classification dataset."""
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    filename = root_url + "FordA_TRAIN.tsv"
    if not train:
        filename = root_url + "FordA_TEST.tsv"
    data = pd.read_csv(filename, sep="\t")
    y = data.values[:, 0].astype(int)
    x = data.values[:, 1:]
    y[y == -1] = 0
    return np.expand_dims(x, -1), y
