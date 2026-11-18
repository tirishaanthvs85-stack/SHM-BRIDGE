from sklearn.svm import SVR


def build_model():

    model = SVR(
        kernel="rbf",
        C=100.0,
        epsilon=0.001,
        gamma="scale"
    )

    return model