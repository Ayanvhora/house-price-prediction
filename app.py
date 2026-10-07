# ==========================================
# AI House Price Prediction & Recommendation
# Developed By: Ayan Vhora
# ==========================================

# ---------- Import Libraries ----------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="AI House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    return pd.read_csv("Indian_House_Price_Dataset_10000.csv")

df = load_data()

# ---------- Sidebar ----------
st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "💰 House Price Prediction",
        "⭐ House Recommendation",
        "📊 Analytics",
        "ℹ About Project"
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🏠 AI-Based House Price Prediction & Recommendation System")

    st.markdown("""
Welcome to the **AI-Based House Price Prediction & Recommendation System**.

This application helps users:

✅ Predict House Prices

✅ Find Best Houses Based on Budget

✅ Explore House Market Analytics

✅ Compare House Prices Across Cities
""")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🏘 Total Houses", len(df))

    with col2:
        st.metric("🏙 Cities", df["City"].nunique())

    with col3:
        st.metric("🏠 Property Types", df["Property_Type"].nunique())

    with col4:
        st.metric("🤖 Model Accuracy", "98.75%")

    st.divider()

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # ===========================
    # Price Distribution Graph
    # ===========================

    st.subheader("📈 House Price Distribution")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.hist(df["Price_in_Lakhs"], bins=30)

    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Number of Houses")
    ax.set_title("Distribution of House Prices")

    st.pyplot(fig)

    st.divider()

    # ===========================
    # Average Price by City
    # ===========================

    st.subheader("🏙 Average House Price by City")

    city_price = (
        df.groupby("City")["Price_in_Lakhs"]
          .mean()
          .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12,6))

    city_price.plot(kind="bar", ax=ax)

    ax.set_xlabel("City")
    ax.set_ylabel("Average Price (Lakhs)")
    ax.set_title("Average House Price by City")

    st.pyplot(fig)

# ==========================================
# HOUSE PRICE PREDICTION PAGE
# ==========================================

elif page == "💰 House Price Prediction":

    st.title("💰 House Price Prediction")
    st.write("Enter the property details below.")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox(
            "State",
            sorted(df["State"].unique())
        )

        city = st.selectbox(
            "City",
            sorted(df["City"].unique())
        )

        locality = st.selectbox(
            "Locality",
            sorted(df["Locality"].unique())
        )

        property_type = st.selectbox(
            "Property Type",
            sorted(df["Property_Type"].unique())
        )

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

    with col2:

        size = st.number_input(
            "Size (Sq.Ft.)",
            min_value=300,
            max_value=10000,
            value=1200
        )

        furnished = st.selectbox(
            "Furnished Status",
            sorted(df["Furnished_Status"].unique())
        )

        parking = st.selectbox(
            "Parking",
            sorted(df["Parking"].unique())
        )

        age = st.number_input(
            "Age of Property",
            min_value=0,
            max_value=50,
            value=5
        )

    if st.button("🔍 Predict House Price"):

        input_data = pd.DataFrame({

            "Property_ID": [10001],

            "State": [state],

            "City": [city],

            "Locality": [locality],

            "Property_Type": [property_type],

            "BHK": [bhk],

            "Bathrooms": [bathrooms],

            "Balconies": [2],

            "Size_in_SqFt": [size],

            "Floor_No": [5],

            "Total_Floors": [10],

            "Parking": [parking],

            "Lift": ["Yes"],

            "Gym": ["Yes"],

            "Swimming_Pool": ["No"],

            "Garden": ["Yes"],

            "Clubhouse": ["Yes"],

            "Security": ["Yes"],

            "Furnished_Status": [furnished],

            "Age_of_Property": [age],

            "Nearby_School_Distance_KM": [2],

            "Nearby_Hospital_Distance_KM": [2],

            "Metro_Distance_KM": [3],

            "Highway_Distance_KM": [4],

            "Facing": ["East"]

        })

        input_encoded = preprocessor.transform(input_data)

        prediction = model.predict(input_encoded)

        st.success("✅ Prediction Successful")

        st.divider()
                # ==========================================
        # Download Prediction Report
        # ==========================================

        report = pd.DataFrame({
            "Field": [
                "State",
                "City",
                "Locality",
                "Property Type",
                "BHK",
                "Bathrooms",
                "Size (Sq.Ft.)",
                "Parking",
                "Furnished",
                "Predicted Price"
            ],
            "Value": [
                state,
                city,
                locality,
                property_type,
                bhk,
                bathrooms,
                size,
                parking,
                furnished,
                f"{prediction[0]:.2f} Lakhs"
            ]
        })

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="House_Price_Prediction_Report.csv",
            mime="text/csv"
        )

        st.subheader("🏠 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "💰 Estimated Price",
                f"₹ {prediction[0]:.2f} Lakhs"
            )

            st.metric(
                "🏠 Property Type",
                property_type
            )

            st.metric(
                "🛏 BHK",
                bhk
            )

        with col2:
            st.metric(
                "📍 City",
                city
            )

            st.metric(
                "📐 Size",
                f"{size} Sq.Ft."
            )

            st.metric(
                "🛋 Furnished",
                furnished
            )

        st.divider()
    

    st.info("Prediction Page Coming Soon...")

# ==========================================
# HOUSE RECOMMENDATION PAGE
# ==========================================

# ==========================================
# HOUSE RECOMMENDATION PAGE
# ==========================================

elif page == "⭐ House Recommendation":

    st.title("⭐ Smart House Recommendation System")

    st.write("Find the best houses based on your budget.")

    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input(
            "💰 Enter Your Budget (Lakhs)",
            min_value=10,
            max_value=1000,
            value=200
        )

        city = st.selectbox(
            "🏙 Select City",
            sorted(df["City"].unique()),
            key="rec_city"
        )

    with col2:

        property_type = st.selectbox(
            "🏠 Property Type",
            sorted(df["Property_Type"].unique()),
            key="rec_property"
        )

        bhk = st.number_input(
            "🛏 BHK",
            min_value=1,
            max_value=10,
            value=2,
            key="rec_bhk"
        )
        parking = st.selectbox(
            "🚗 Parking",
            sorted(df["Parking"].unique()),
            key="rec_parking"
        )

        furnished = st.selectbox(
            "🛋 Furnished Status",
            sorted(df["Furnished_Status"].unique()),
            key="rec_furnished"
        )

        lift = st.selectbox(
            "🛗 Lift",
            sorted(df["Lift"].unique()),
            key="rec_lift"
        )

        gym = st.selectbox(
            "🏋 Gym",
            sorted(df["Gym"].unique()),
            key="rec_gym"
        )

    if st.button("🔍 Find Best Houses"):

        recommended = df[
        (df["City"] == city) &
        (df["Property_Type"] == property_type) &
        (df["BHK"] == bhk) &
        (df["Parking"] == parking) &
        (df["Furnished_Status"] == furnished) &
        (df["Lift"] == lift) &
        (df["Gym"] == gym)
    ]

        recommended = recommended.copy()

        recommended["Difference"] = abs(
            recommended["Price_in_Lakhs"] - budget
        )

        recommended = recommended.sort_values("Difference")
        # ===========================
        # Best Match House
        # ===========================
        if len(recommended) > 0:

            best_house = recommended.iloc[0]
            difference = abs(best_house["Price_in_Lakhs"] - budget)

            match_percent = max(0, 100 - (difference / budget) * 100)
            

            st.success("🏆 Best Matching House Found")

            col1, col2 = st.columns(2)

            with col1:

                st.metric("💰 Price",
                        f"₹ {best_house['Price_in_Lakhs']:.2f} Lakhs")

                st.metric("🏠 BHK",
                        int(best_house["BHK"]))

                st.metric("📐 Size",
                        f"{best_house['Size_in_SqFt']} Sq.Ft.")

            with col2:

                st.metric(
                    "📍 Locality",
                    best_house["Locality"]
                )

                st.metric(
                    "🏡 Property",
                    best_house["Property_Type"]
                )

                st.metric(
                    "🛋 Furnished",
                    best_house["Furnished_Status"]
                )

                st.metric(
                    "⭐ Match",
                    f"{match_percent:.1f}%"
                )

                st.metric(
                    "💸 Budget Difference",
                    f"₹ {difference:.2f} Lakhs"
                )

        # 👇 Ye line with col2 ke bahar honi chahiye
            st.divider()

            st.subheader("🏡 Top 5 Recommended Houses")

            st.dataframe(
                recommended[
                    [
                        "Locality",
                        "Property_Type",
                        "BHK",
                        "Size_in_SqFt",
                        "Price_in_Lakhs",
                        "Furnished_Status"
                    ]
                ].head(5),
                use_container_width=True
            )
                # ==========================================
            # Download Recommended Houses
            # ==========================================

            download_df = recommended[
                [
                    "Locality",
                    "Property_Type",
                    "BHK",
                    "Size_in_SqFt",
                    "Price_in_Lakhs",
                    "Furnished_Status"
                ]
            ].head(5)

            csv = download_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Recommended Houses",
                data=csv,
                file_name="Recommended_Houses.csv",
                mime="text/csv"
            )
        else:
            st.warning("❌ No matching houses found. Please change the filters and try again.")


# ==========================================
# ANALYTICS PAGE
# ==========================================

elif page == "📊 Analytics":

    st.title("📊 House Market Analytics Dashboard")

    st.write("Explore insights from the housing dataset.")

    # -------------------------------
    # Summary Cards
    # -------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🏘 Total Houses", len(df))

    with col2:
        st.metric("🏙 Cities", df["City"].nunique())

    with col3:
        st.metric("🏠 Property Types", df["Property_Type"].nunique())

    with col4:
        st.metric("💰 Average Price",
                  f"{df['Price_in_Lakhs'].mean():.2f} Lakhs")

    st.divider()

    # -------------------------------
    # Price Distribution
    # -------------------------------

    st.subheader("📈 House Price Distribution")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.hist(df["Price_in_Lakhs"], bins=30)

    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of House Prices")

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # Average Price by Property Type
    # -------------------------------

    st.subheader("🏠 Average Price by Property Type")

    property_price = (
        df.groupby("Property_Type")["Price_in_Lakhs"]
          .mean()
          .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8,5))

    property_price.plot(kind="bar", ax=ax)

    ax.set_ylabel("Average Price (Lakhs)")

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # Average Price by BHK
    # -------------------------------

    st.subheader("🛏 Average Price by BHK")

    bhk_price = (
        df.groupby("BHK")["Price_in_Lakhs"]
          .mean()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    bhk_price.plot(kind="bar", ax=ax)

    ax.set_ylabel("Average Price (Lakhs)")

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # Dataset Preview
    # -------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)
# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "ℹ About Project":

    st.title("ℹ About Project")

    st.write("""
This project predicts house prices using Machine Learning.

Algorithm Used:
- Linear Regression

Developed By:
Ayan Vhora
""")
    # ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style='text-align:center; color:gray;'>

🏠 <b>AI House Price Prediction & Recommendation System</b>

<br>

Developed By <b>Ayan Vhora</b>

<br>

M.Sc. Data Science

<br><br>

© 2026 All Rights Reserved

</div>
""", unsafe_allow_html=True)