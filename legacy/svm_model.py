from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_svm_model():

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("svr", SVR(
            kernel="rbf",
            C=100.0,
            gamma="scale",
            epsilon=0.001
        ))
    ])

    return model