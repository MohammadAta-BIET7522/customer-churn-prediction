import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SAMPLE DATA ----------------
@st.cache_data
def load_sample_data():
    data = {
        'Age':[45,38,47,58,37,29,52,41,35,44,33,55,26,60,48],
        'Gender':['Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male'],
        'Tenure':[43,1,45,51,24,3,60,12,6,36,18,48,9,54,30],
        'Usage Frequency':[20,18,18,2,4,15,22,8,12,19,10,25,7,5,16],
        'Support Calls':[4,10,7,2,4,5,1,9,3,2,6,1,8,3,2],
        'Payment Delay':[24,12,26,5,30,15,0,22,18,7,25,3,28,10,14],
        'Subscription Type':['Standard','Standard','Basic','Standard','Standard','Premium','Premium','Basic','Standard','Premium','Basic','Premium','Standard','Basic','Premium'],
        'Contract Length':['Annual','Annual','Quarterly','Annual','Quarterly','Monthly','Annual','Quarterly','Monthly','Annual','Quarterly','Annual','Monthly','Annual','Quarterly'],
        'Total Spend':[127,859,514,256,220,450,999,312,185,789,345,876,234,765,543],
        'Churn':[0,1,0,0,1,0,0,1,1,0,1,0,1,0,0]
    }
    return pd.DataFrame(data)

# ---------------- PREPROCESS ----------------
def preprocess_data(df):
    df = df.copy()

    df['Gender'] = df['Gender'].map({'Male':0,'Female':1,'Other':2}).fillna(0)
    df['Subscription Type'] = df['Subscription Type'].map({'Basic':0,'Standard':1,'Premium':2}).fillna(0)
    df['Contract Length'] = df['Contract Length'].map({'Monthly':0,'Quarterly':1,'Annual':2}).fillna(0)

    return df

# ---------------- MODEL ----------------
@st.cache_resource
def train_model(X, y, model_type):
    if model_type == "Random Forest":
        model = RandomForestClassifier()
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = SVC(probability=True)

    model.fit(X, y)
    return model

# ---------------- NAVIGATION ----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", [
    "Data Overview",
    "EDA",
    "Preprocessing",
    "Model Training",
    "Batch Prediction",
    "Predict Single"
])

st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)

df = load_sample_data()

# ---------------- OVERVIEW ----------------
if page == "Data Overview":
    st.subheader("Dataset")
    st.dataframe(df.head())

# ---------------- EDA ----------------
elif page == "EDA":
    st.subheader("Churn Distribution")
    st.bar_chart(df['Churn'].value_counts())

    st.subheader("Correlation")
    numeric_df = df.select_dtypes(include=np.number)
    st.write(numeric_df.corr())

# ---------------- PREPROCESS ----------------
elif page == "Preprocessing":
    processed = preprocess_data(df)
    st.dataframe(processed.head())

# ---------------- TRAIN ----------------
elif page == "Model Training":
    processed = preprocess_data(df)

    X = processed.drop("Churn", axis=1)
    y = processed["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_type = st.selectbox("Select Model", ["Random Forest","Logistic Regression","SVM"])

    if st.button("Train Model"):
        model = train_model(X_train, y_train, model_type)

        y_pred = model.predict(X_test)

        st.success("Model Trained!")

        col1,col2,col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test,y_pred):.2%}")
        col2.metric("Precision", f"{precision_score(y_test,y_pred):.2%}")
        col3.metric("Recall", f"{recall_score(y_test,y_pred):.2%}")

# ---------------- BATCH ----------------
elif page == "Batch Prediction":
    file = st.file_uploader("Upload CSV")

    if file is not None:
        batch = pd.read_csv(file)
        processed_batch = preprocess_data(batch)

        processed = preprocess_data(df)
        X = processed.drop("Churn", axis=1)
        y = processed["Churn"]

        model = train_model(X, y, "Random Forest")

        pred = model.predict(processed_batch)
        batch["Prediction"] = pred

        st.dataframe(batch)

# ---------------- SINGLE ----------------
elif page == "Predict Single":

    age = st.slider("Age",18,100,30)
    gender = st.selectbox("Gender",["Male","Female","Other"])
    tenure = st.slider("Tenure",0,72,12)
    usage = st.slider("Usage Frequency",0,30,10)
    support = st.slider("Support Calls",0,20,2)
    delay = st.slider("Payment Delay",0,60,5)
    sub = st.selectbox("Subscription Type",["Basic","Standard","Premium"])
    contract = st.selectbox("Contract Length",["Monthly","Quarterly","Annual"])
    spend = st.slider("Total Spend",0,2000,500)

    if st.button("Predict"):

        user = pd.DataFrame({
            'Age':[age],
            'Gender':[gender],
            'Tenure':[tenure],
            'Usage Frequency':[usage],
            'Support Calls':[support],
            'Payment Delay':[delay],
            'Subscription Type':[sub],
            'Contract Length':[contract],
            'Total Spend':[spend]
        })

        processed_user = preprocess_data(user)

        processed = preprocess_data(df)
        X = processed.drop("Churn", axis=1)
        y = processed["Churn"]

        model = train_model(X, y, "Random Forest")

        pred = model.predict(processed_user)

        if pred[0] == 1:
            st.error("⚠️ High Churn Risk")
        else:
            st.success("✅ Low Churn Risk")