import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path, odor):
    """
    Load data from the specified file, remove columns with all-zero values,
    normalize features, and save the cleaned data.

    Args:
        file_path (str): Path to the input Excel file.

    Returns:
        pd.DataFrame: Cleaned and normalized data.
    """
    df = pd.read_excel(file_path)
    # Identify columns with all zero values except the last two columns
    cols_to_drop = df.columns[:-2][(df.iloc[:, :-2] == 0).all()]
    # Drop identified columns
    df_cleaned = df.drop(columns=cols_to_drop)
    # Normalize features except the last two columns
    columns_to_normalize = df_cleaned.columns[:-2]
    scaler = StandardScaler()
    df_cleaned[columns_to_normalize] = scaler.fit_transform(df_cleaned[columns_to_normalize])
    # Save cleaned data to a new Excel file
    df_cleaned.to_excel(f'./data/{odor}_cleaned.xlsx', index=False)
    return df_cleaned