import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor



def height_to_inches(h):
    try:
        feet, inches = h.split('-')
        return int(feet) * 12 + int(inches)
    except:
        return np.nan
        

def plot_player_position_distribution(df):
    fig = px.bar(
        df['player_position'].value_counts().reset_index(),
        x='player_position',
        y='count',
        title='Player Position Distribution',
        labels={'player_position': 'Position', 'count': 'Count'},
        color_discrete_sequence=['#FFC000']
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )
    
    fig.show()
    

def plot_play_direction_distribution(df):
    fig = px.bar(
        df['play_direction'].value_counts().reset_index(),
        x='play_direction',
        y='count',
        title='Play Direction Distribution',
        labels={'play_direction': 'Direction', 'count': 'Count'},
        color_discrete_sequence=['#FFC000']
    
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )
            
    fig.show()
    

def plot_numeric_distribution(df, bins=30):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    rows = (len(numeric_cols) + 2) // 3

    fig = sp.make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=numeric_cols
    )

    row, col = 1, 1
    for col_name in numeric_cols:
        fig.add_trace(
            go.Histogram(
                x=df[col_name],
                name=col_name,
                marker=dict(color='#FFC000'),
                nbinsx=bins
            ),
            row=row, col=col
        )

        col += 1
        if col > 3:
            col = 1
            row += 1

    fig.update_layout(
        height=300 * rows,
        title_text="Distribution of Numeric Variables",
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )

    fig.update_layout(margin=dict(l=40, r=40, t=80, b=40))

    fig.show()
    

def plot_numeric_boxplots(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    rows = int(np.ceil(len(numeric_cols) / 3))
    cols = 3

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=numeric_cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    row = 1
    col = 1

    for col_name in numeric_cols:
        fig.add_trace(
            go.Box(
                y=df[col_name],
                name=col_name,
                marker_color="#FFC000"
            ),
            row=row,
            col=col
        )

        col += 1
        if col > cols:
            col = 1
            row += 1

    fig.update_layout(
        height=300 * rows,
        title_text="Box Plots of Numeric Variables",
        showlegend=False,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()


def plot_corr_matrix(df_cleaned):
    corr = df_cleaned.corr(numeric_only=True)
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale="solar",
    )
    
    fig.update_layout(
        title="Correlation Heatmap",
        width=1000,
        height=900,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        coloraxis_colorbar=dict(
            title=dict(
                text="Corr",
                font=dict(color="white")
            ),
            tickfont=dict(color="white")
        )
    )
    
    fig.update_xaxes(tickangle=90)
    fig.update_yaxes(automargin=True)
    
    fig.show()
    

def process_outliers(df, cols):
    df = df.copy()
    for col in cols:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        
        iqr = q3 - q1
        
        lower = q1 - (1.5*iqr)
        upper = q3 + (1.5*iqr)

        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {"MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2}
    

def plot_residuals(y_true, y_pred, title="Residuals vs Predicted"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    ax.set_facecolor("black")
    plt.gcf().patch.set_facecolor("black")

    ax.scatter(y_pred, residuals, alpha=0.2, s=1, color="#FFC000")
    ax.axhline(0, color="white", linestyle="--", linewidth=1.2)

    ax.set_title(title, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Residual", color="white")

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.show()
    

def rf_grid_search(X_train, y_train, X_test, y_test, param_grid, cv=5, target_name=""):
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    preds = best_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n===== Random Forest Grid Search ({target_name}) =====")
    print("Best Params:", grid_search.best_params_)
    print(f"MAE : {mae:0.4f}")
    print(f"RMSE: {rmse:0.4f}")
    print(f"MSE : {mse:0.4f}")
    print(f"R²  : {r2:0.4f}")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2
    }

    return best_model, preds, grid_search, metrics
    

def plot_feature_importances(model, feature_names, top_n=15, title="Feature Importances"):
    importances = np.asarray(model.feature_importances_)
    feature_names = np.asarray(feature_names)

    imp_series = pd.Series(importances, index=feature_names)
    imp_series = imp_series.sort_values(ascending=False).head(top_n)[::-1]

    plt.figure(figsize=(8, max(4, top_n * 0.4)))
    ax = plt.gca()

    ax.set_facecolor("black")
    plt.gcf().patch.set_facecolor("black")

    ax.barh(imp_series.index, imp_series.values, color="#FFC000")

    ax.set_title(title, color="white", pad=12)
    ax.set_xlabel("Importance", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    plt.show()


def xgb_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid,
    cv=5,
    target_name=""
):

    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        eval_metric="rmse"
    )

    grid = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n===== XGBoost Grid Search ({target_name}) =====")
    print("Best Params:", grid.best_params_)
    print(f"MAE : {mae:0.4f}")
    print(f"RMSE: {rmse:0.4f}")
    print(f"MSE : {mse:0.4f}")
    print(f"R²  : {r2:0.4f}")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2
    }

    return best_model, preds, grid, metrics

def pairplot(df, sample_size=1000, random_state=42):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    sample_df = df[numeric_cols].sample(sample_size, random_state=random_state)

    sns.set_style("dark")
    sns.set_context("notebook")
    plt.figure(figsize=(12, 12))

    g = sns.pairplot(
        sample_df,
        diag_kind="kde",
        corner=True,
        plot_kws=dict(alpha=0.3, color="#FFC000"),
        diag_kws=dict(color="#FFC000")
    )

    g.fig.patch.set_facecolor("black")
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

    plt.show()
