from data_processing import load_and_preprocess_data
from model_training import train_models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


def main():
    """
    Main function to load data, preprocess it, and train multiple models
    using cross-validation strategies.
    """
    odor = 'fish'
    file_path = f'./data/{odor}.xlsx'
    df_cleaned = load_and_preprocess_data(file_path, odor)
    X = df_cleaned.iloc[:, :-2].to_numpy()
    y = df_cleaned.iloc[:, -1].to_numpy()
    columns = df_cleaned.columns.tolist()

    # Define cross-validation strategies
    cv_list = [(3, '3-Fold'), (5, '5-Fold')]

    # Define models to train
    models = [
        KNeighborsClassifier(n_neighbors=5),
        GaussianNB(),
        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        SVC(kernel='linear', random_state=42, probability=True),
        SGDClassifier(loss='log_loss', random_state=42),
        GradientBoostingClassifier(n_estimators=50, max_depth=15, learning_rate=0.5, random_state=42)
    ]

    # Initialize dictionary to store accuracy scores
    acc_lists = {f'accuracy_list_{i[0]}': [] for i in cv_list}

    # Train models and evaluate performance
    train_models(X, y, columns, models, acc_lists, cv_list, odor)


if __name__ == '__main__':
    main()
