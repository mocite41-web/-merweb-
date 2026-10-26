import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Facility Reservation Predictive System", page_icon="🏫", layout="wide")

REQUIRED = ['Reservation_ID','User_Type','Facility_Type','Days_In_Advance','Participants','Usage_Status','Actual_Attendance','Cancellation_Days_Before']
PREDICTORS = ['User_Type','Facility_Type','Days_In_Advance','Participants','Lead_Time_Category']

def lead_category(days):
    if days <= 2: return 'Last Minute'
    if days <= 7: return 'Short Lead'
    if days <= 14: return 'Medium Lead'
    return 'Early Booking'

@st.cache_data
def clean_data(df):
    out = df.copy()
    out = out.drop_duplicates(subset=['Reservation_ID'])
    out['Days_In_Advance'] = pd.to_numeric(out['Days_In_Advance'], errors='coerce').clip(lower=0)
    out['Participants'] = pd.to_numeric(out['Participants'], errors='coerce')
    out['Actual_Attendance'] = pd.to_numeric(out['Actual_Attendance'], errors='coerce')
    out['Days_In_Advance'] = out['Days_In_Advance'].fillna(out['Days_In_Advance'].median())
    out['Participants'] = out['Participants'].fillna(out['Participants'].median())
    out['Actual_Attendance'] = out['Actual_Attendance'].fillna(0)
    out['Lead_Time_Category'] = out['Days_In_Advance'].apply(lead_category)
    out['Reservation_Outcome'] = out['Usage_Status'].apply(lambda x: 'Completed' if str(x).strip().lower() == 'used' else 'Not Completed')
    out['Attendance_Ratio'] = np.where(out['Participants'] == 0, 0.0, out['Actual_Attendance'] / out['Participants'])
    return out

@st.cache_resource
def train_model(df_clean):
    X = df_clean[PREDICTORS].copy()
    y = df_clean['Reservation_Outcome'].copy()
    X_enc = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=.20, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, pos_label='Not Completed', zero_division=0),
        'Recall': recall_score(y_test, pred, pos_label='Not Completed', zero_division=0),
        'F1-Score': f1_score(y_test, pred, pos_label='Not Completed', zero_division=0),
    }
    cm = confusion_matrix(y_test, pred, labels=['Completed','Not Completed'])
    importances = pd.Series(model.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
    return model, X_enc.columns.tolist(), metrics, cm, importances

@st.cache_data
def load_default():
    return pd.read_csv('facility_reservations.csv')

st.sidebar.title("🏫 Facility Reservation")
page = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Decision-support system for cancellation/no-show risk")

try:
    default_df = load_default()
except Exception as e:
    st.error(f"Default dataset could not be loaded: {e}")
    st.stop()

uploaded = st.sidebar.file_uploader("Optional: Upload another CSV", type=['csv'])
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    raw = default_df

missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

clean = clean_data(raw)
model, model_columns, metrics, cm, importances = train_model(clean)

if page == "Home":
    st.title("Campus Facility Reservation Monitoring System")
    st.write("This Streamlit application reuses the midterm facility-reservation dataset, cleaning steps, engineered features, and Decision Tree model to support administrator review of possible cancellation or no-show risk.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Reservations", f"{len(clean):,}")
    c2.metric("Completed", int((clean.Reservation_Outcome=='Completed').sum()))
    c3.metric("Not Completed", int((clean.Reservation_Outcome=='Not Completed').sum()))
    c4.metric("Model Accuracy", f"{metrics['Accuracy']:.1%}")
    st.info("Predictions are decision-support outputs. Final reservation decisions should remain with an authorized facility administrator.")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    st.success("Dataset schema is valid.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows (raw)", len(raw)); c2.metric("Rows (clean)", len(clean))
    c3.metric("Duplicate IDs", int(raw.duplicated(subset=['Reservation_ID']).sum()))
    c4.metric("Missing Values", int(raw.isna().sum().sum()))
    st.subheader("Data Preview")
    st.dataframe(clean.head(20), use_container_width=True)
    st.subheader("Null Summary")
    nulls = pd.DataFrame({'Column': raw.columns, 'Null_Count': raw.isna().sum().values})
    st.dataframe(nulls, use_container_width=True)
    st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode('utf-8'), "cleaned_facility_reservations.csv", "text/csv")

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    st.subheader("Descriptive Statistics")
    st.dataframe(clean.describe(include='all').transpose(), use_container_width=True)
    st.subheader("Interactive Filters")
    facility = st.multiselect("Facility Type", sorted(clean.Facility_Type.dropna().unique()), default=sorted(clean.Facility_Type.dropna().unique()))
    filtered = clean[clean.Facility_Type.isin(facility)]
    c1,c2,c3 = st.columns(3)
    with c1:
        st.caption("Reservations by Facility")
        st.bar_chart(filtered['Facility_Type'].value_counts())
    with c2:
        st.caption("Outcome Distribution")
        st.bar_chart(filtered['Reservation_Outcome'].value_counts())
    with c3:
        st.caption("Lead-Time Categories")
        st.bar_chart(filtered['Lead_Time_Category'].value_counts())
    st.subheader("Outcome by User Type (%)")
    user_ct = pd.crosstab(filtered.User_Type, filtered.Reservation_Outcome, normalize='index') * 100
    st.bar_chart(user_ct)

elif page == "Prediction":
    st.title("Predict Cancellation / No-Show Risk")
    st.write("Enter only information available before the reservation outcome is known.")
    c1,c2 = st.columns(2)
    with c1:
        user_type = st.selectbox("User Type", sorted(clean.User_Type.dropna().unique()))
        facility_type = st.selectbox("Facility Type", sorted(clean.Facility_Type.dropna().unique()))
    with c2:
        days = st.number_input("Days in Advance", min_value=0, max_value=365, value=7)
        participants = st.number_input("Participants", min_value=1, max_value=10000, value=30)
    category = lead_category(days)
    st.caption(f"Lead-Time Category: {category}")
    if st.button("Predict Outcome", type="primary"):
        row = pd.DataFrame([{'User_Type':user_type,'Facility_Type':facility_type,'Days_In_Advance':days,'Participants':participants,'Lead_Time_Category':category}])
        row_enc = pd.get_dummies(row).reindex(columns=model_columns, fill_value=0)
        result = model.predict(row_enc)[0]
        st.subheader(f"Prediction: {result}")
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(row_enc)[0]
            prob_map = dict(zip(model.classes_, probs))
            risk = prob_map.get('Not Completed', 0.0)
            st.metric("Predicted non-completion risk", f"{risk:.1%}")
            if risk >= .60: st.warning("High review priority: administrator follow-up is recommended.")
            elif risk >= .40: st.info("Moderate review priority: consider confirmation follow-up.")
            else: st.success("Lower predicted risk. Continue normal review workflow.")
        st.caption("This prediction does not automatically approve, reject, or cancel a reservation.")

elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    noncomp = (clean.Reservation_Outcome=='Not Completed').mean()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Non-completion Rate", f"{noncomp:.1%}")
    c2.metric("Avg. Lead Time", f"{clean.Days_In_Advance.mean():.1f} days")
    c3.metric("Avg. Participants", f"{clean.Participants.mean():.1f}")
    c4.metric("Facilities", clean.Facility_Type.nunique())
    st.subheader("Model Performance")
    cols = st.columns(4)
    for col,(name,val) in zip(cols, metrics.items()): col.metric(name, f"{val:.1%}")
    st.subheader("Feature Importance")
    st.bar_chart(importances.head(10))
    st.subheader("Review Queue: historical non-completed reservations")
    queue_cols = ['Reservation_ID','User_Type','Facility_Type','Days_In_Advance','Participants','Usage_Status','Lead_Time_Category']
    st.dataframe(clean.loc[clean.Reservation_Outcome=='Not Completed', queue_cols].head(50), use_container_width=True)
    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=['Actual Completed','Actual Not Completed'], columns=['Pred Completed','Pred Not Completed']))

elif page == "System Design":
    st.title("SAD Integration")
    st.subheader("Actors")
    st.write("• Facility Administrator — reviews dashboard, prediction results, and records the final human decision.\n\n• Requesting User — provides reservation information.\n\n• Predictive System — validates data and provides decision-support risk output.")
    st.subheader("Functional Requirements")
    req = pd.DataFrame([
        ['FR-DM-01','The system shall flag reservations made with fewer than 3 days lead time for review and confirmation.'],
        ['FR-DM-02','The system shall support additional review for student/external reservations involving high-demand facilities.'],
        ['FR-DM-03','The system shall provide administrators a dashboard of reservation utilization and non-completion risk.'],
        ['FR-DM-04','The system shall validate uploaded CSV files and report missing required columns.'],
        ['FR-DM-05','The system shall accept pre-reservation inputs and display a predicted outcome for authorized human review.'],
    ], columns=['ID','Requirement'])
    st.dataframe(req, hide_index=True, use_container_width=True)
    st.subheader("Non-Functional Requirements")
    st.write("1. **Usability:** Pages and inputs should be understandable to facility administrators.\n\n2. **Performance:** Common dashboard and prediction actions should respond quickly for the laboratory dataset.\n\n3. **Security/Privacy:** Do not publish sensitive student records or credentials; use authorized or synthetic data for public deployment.")
    st.subheader("Analytical Decision Workflow")
    st.code("Reservation data → Validate → Clean/Engineer features → Predict risk → Dashboard/Review queue → Administrator reviews evidence → Human final decision", language=None)
    st.subheader("Simple Use Case View")
    st.code("Requesting User → [Submit reservation]\nFacility Administrator → [Upload/View data] → [Run prediction] → [Review dashboard] → [Record final decision]\nPredictive System → [Validate data] → [Generate risk prediction]", language=None)
