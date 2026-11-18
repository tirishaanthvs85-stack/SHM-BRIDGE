"""
rf_model.py
Random Forest Regressor Model Definition
Bridge Structural Health Monitoring
"""

from sklearn.ensemble import RandomForestRegressor


def build_rf_model(n_estimators=200, random_state=42):
    """
    Builds and returns a Random Forest Regressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )
    return model
