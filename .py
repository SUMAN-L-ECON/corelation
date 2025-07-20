import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from io import StringIO

st.set_page_config(page_title="Multiple Regression Explorer", layout="wide")

st.title("📊 Multiple Regression Analysis App")

# Upload Section
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select 2 or more numeric columns for Regression", options=all_columns)

    if len(selected_columns) >= 2:
        df_selected = df[selected_columns].copy()

        # Convert non-numeric to numeric
        for col in df_selected.columns:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

        df_selected.dropna(inplace=True)

        st.subheader("Cleaned Data Used for Regression")
        st.dataframe(df_selected)

        # Correlation Heatmap
        st.subheader("📌 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df_selected.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Regression Model (statsmodels for p-values)
        y = df_selected.iloc[:, 0]  # first selected column as dependent
        X = df_selected.iloc[:, 1:]  # rest as independent
        X_const = sm.add_constant(X)

        model = sm.OLS(y, X_const).fit()
        st.subheader("📄 Regression Summary (statsmodels)")
        st.text(model.summary())

        # Advanced Visualizations
        st.subheader("🔍 Pairplot of Selected Variables")
        st.pyplot(sns.pairplot(df_selected, diag_kind='kde'))

        st.subheader("📈 Residual Plot")
        pred = model.predict(X_const)
        residuals = y - pred
        fig2, ax2 = plt.subplots()
        sns.residplot(x=pred, y=residuals, lowess=True, ax=ax2, line_kws={"color": "red"})
        ax2.set_xlabel("Fitted values")
        ax2.set_ylabel("Residuals")
        st.pyplot(fig2)

        st.subheader("🎯 Actual vs Predicted")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y, pred, edgecolors='k')
        ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        st.pyplot(fig3)

        # Significance Report
        st.subheader("⭐ Significance Report")
        signif_df = pd.DataFrame({
            "Variable": X_const.columns,
            "Coefficient": model.params,
            "P-value": model.pvalues,
            "Significance": model.pvalues.apply(lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns")
        })
        st.dataframe(signif_df)

        st.success("✅ Regression completed and visualizations generated.")
    else:
        st.warning("Please select **at least 2 numeric columns** for regression.")
else:
    st.info("👆 Upload a dataset to begin.")
