import pandas as pd
from pipelines import separate_column_types
from cleaners import clean_categorical


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    _, categorical_columns = separate_column_types(data)

    cleaned_cat_data = clean_categorical(X=data, columns=categorical_columns)

    return cleaned_cat_data