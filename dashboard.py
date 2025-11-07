import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 🚦 Traffic Violation Prediction Dashboard
# =========================================================

# Page configuration
st.set_page_config(
    page_title="Traffic Violation Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# Load Data & Models
# =========================================================

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv("Indian_Traffic_Violations.csv")
        # Apply violation mapping
        def map_violation_type(val):
            if pd.isna(val): return "Others"
            s = str(val).lower()
            if "speed" in s or "over-speed" in s or "overspeed" in s:
                return "Over-Speed"
            if "drunk" in s or "alcohol" in s:
                return "Drunken Driving"
            if "wrong" in s and "side" in s:
                return "Driving on Wrong Side"
            if "signal" in s or "red" in s or "jump" in s:
                return "Jumping Red Light / Signal"
            if "mobile" in s or "phone" in s or "distract" in s:
                return "Use of Mobile / Distracted Driving"
            return "Others"
        
        df['target_group'] = df['Violation_Type'].apply(map_violation_type)
        
        # Date features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['month'] = df['Date'].dt.month.fillna(0).astype(int)
            df['dayofweek'] = df['Date'].dt.dayofweek.fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    try:
        gb_model = joblib.load("model_gradient_boosting_tuned.joblib")
        le_target = joblib.load("label_encoder_target.joblib")
        label_encoders = joblib.load("label_encoders.joblib")
        imputer = joblib.load("imputer.joblib")
        return gb_model, le_target, label_encoders, imputer
    except Exception as e:
        st.warning(f"Models not found: {e}")
        return None, None, None, None

# Load data
df = load_data()
gb_model, le_target, label_encoders, imputer = load_models()

# =========================================================
# Sidebar Navigation
# =========================================================

st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Dashboard Overview", "📈 Data Exploration", "🤖 Model Performance", "🔮 Make Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard:**
- Analyze 100K traffic violations
- Explore patterns and trends
- Predict violation types
- Compare model performance
""")

# =========================================================
# PAGE 1: Dashboard Overview
# =========================================================

if page == "📊 Dashboard Overview":
    st.markdown('<p class="main-header">🚦 Traffic Violation Analytics Dashboard</p>', unsafe_allow_html=True)
    
    if df is not None:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Violations", f"{len(df):,}")
        with col2:
            st.metric("Violation Categories", df['target_group'].nunique())
        with col3:
            avg_fine = df['Fine_Amount'].mean()
            st.metric("Avg Fine Amount", f"₹{avg_fine:,.0f}")
        with col4:
            total_fines = df['Fine_Amount'].sum()
            st.metric("Total Fines Collected", f"₹{total_fines/1e6:.1f}M")
        
        st.markdown("---")
        
        # Two column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Violation Distribution")
            violation_counts = df['target_group'].value_counts()
            fig = px.pie(
                values=violation_counts.values,
                names=violation_counts.index,
                title="Distribution of Violation Types",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 Fine Amount by Violation")
            avg_fine_by_type = df.groupby('target_group')['Fine_Amount'].mean().sort_values(ascending=True)
            fig = px.bar(
                x=avg_fine_by_type.values,
                y=avg_fine_by_type.index,
                orientation='h',
                title="Average Fine Amount by Violation Type",
                labels={'x': 'Average Fine (₹)', 'y': 'Violation Type'},
                color=avg_fine_by_type.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based analysis
        st.subheader("📅 Temporal Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            month_counts = df['month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig = px.line(
                x=[month_names[i-1] if i > 0 else 'Unknown' for i in month_counts.index],
                y=month_counts.values,
                title="Violations by Month",
                labels={'x': 'Month', 'y': 'Number of Violations'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            day_counts = df['dayofweek'].value_counts().sort_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig = px.bar(
                x=[day_names[i] if i < 7 else 'Unknown' for i in day_counts.index],
                y=day_counts.values,
                title="Violations by Day of Week",
                labels={'x': 'Day', 'y': 'Number of Violations'},
                color=day_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 2: Data Exploration
# =========================================================

elif page == "📈 Data Exploration":
    st.markdown('<p class="main-header">📈 Data Exploration & Insights</p>', unsafe_allow_html=True)
    
    if df is not None:
        # Data preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.markdown("---")
        
        # Vehicle Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚗 Top 10 Vehicle Types")
            vehicle_counts = df['Vehicle_Type'].value_counts().head(10)
            fig = px.bar(
                x=vehicle_counts.values,
                y=vehicle_counts.index,
                orientation='h',
                title="Most Common Vehicles in Violations",
                color=vehicle_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎨 Vehicle Colors")
            color_counts = df['Vehicle_Color'].value_counts().head(10)
            fig = px.pie(
                values=color_counts.values,
                names=color_counts.index,
                title="Distribution of Vehicle Colors"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Driver Demographics
        st.subheader("👥 Driver Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='Driver_Age',
                nbins=30,
                title="Driver Age Distribution",
                labels={'Driver_Age': 'Age', 'count': 'Frequency'},
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gender_counts = df['Driver_Gender'].value_counts()
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Driver Gender Distribution",
                color_discrete_sequence=['#4ECDC4', '#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # State Analysis
        st.subheader("🗺️ Registration State Analysis")
        state_counts = df['Registration_State'].value_counts().head(15)
        fig = px.bar(
            x=state_counts.index,
            y=state_counts.values,
            title="Top 15 States by Violations",
            labels={'x': 'State', 'y': 'Number of Violations'},
            color=state_counts.values,
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 3: Model Performance
# =========================================================

elif page == "🤖 Model Performance":
    st.markdown('<p class="main-header">🤖 Model Performance Analysis</p>', unsafe_allow_html=True)
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    
    performance_data = {
        'Model': ['Baseline RF', 'Tuned RF', 'Baseline GB', 'Tuned GB'],
        'Accuracy': [0.563, 0.569, 0.885, 0.785],
        'F1 Score': [0.419, 0.434, 0.882, 0.769]
    }
    perf_df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            perf_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Greens',
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            perf_df,
            x='Model',
            y='F1 Score',
            title="Model F1 Score Comparison",
            color='F1 Score',
            color_continuous_scale='Blues',
            text='F1 Score'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model details
    st.markdown("---")
    st.subheader("🏆 Best Model: Baseline Gradient Boosting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "88.5%", "+0.5%")
    with col2:
        st.metric("Weighted F1", "0.882", "+0.012")
    with col3:
        st.metric("Macro F1", "0.86", "+0.04")
    
    # Classification report
    st.subheader("📋 Detailed Classification Report (Baseline GB)")
    
    report_data = {
        'Class': ['Drunken Driving', 'Jumping Red Light / Signal', 'Others', 'Over-Speed', 'Use of Mobile / Distracted Driving'],
        'Precision': [1.00, 1.00, 0.83, 1.00, 1.00],
        'Recall': [0.72, 0.77, 1.00, 0.70, 0.79],
        'F1-Score': [0.84, 0.87, 0.91, 0.82, 0.88],
        'Support': [2479, 2246, 11005, 2214, 2056]
    }
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True)
    
    # ✅ Feature Importance (Gradient Boosting)
    if gb_model is not None:
        st.markdown("---")
        st.subheader("🎯 Feature Importance (Gradient Boosting)")
        
        features = ['Registration_State', 'Vehicle_Type', 'Fine_Amount', 'Driver_Age', 'month', 'dayofweek']
        importances = gb_model.feature_importances_
        
        fi_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Ranking",
            color='Importance',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    st.info("""
    **Model Performance Summary:**
    - **Gradient Boosting** outperforms Random Forest significantly
    - Baseline GB achieves **88.5% accuracy** (best overall)
    - **"Others"** category has highest recall (100%) but lower precision (83%)
    - Tuning improved Random Forest slightly but decreased GB performance
    - **Recommendation:** Use Baseline Gradient Boosting for production
    """)

# =========================================================
# PAGE 4: Make Prediction
# =========================================================

elif page == "🔮 Make Prediction":
    st.markdown('<p class="main-header">🔮 Traffic Violation Predictor</p>', unsafe_allow_html=True)
    
    if all([gb_model, le_target, label_encoders, imputer]):
        st.info("Fill in the details below to predict the violation type")
        
        col1, col2 = st.columns(2)
        
        with col1:
            registration_state = st.selectbox(
                "Registration State",
                options=sorted(df['Registration_State'].unique()) if df is not None else ['State1']
            )
            
            vehicle_type = st.selectbox(
                "Vehicle Type",
                options=sorted(df['Vehicle_Type'].unique()) if df is not None else ['Car']
            )
            
            fine_amount = st.number_input(
                "Fine Amount (₹)",
                min_value=0,
                max_value=10000,
                value=2500,
                step=100
            )
        
        with col2:
            driver_age = st.slider(
                "Driver Age",
                min_value=18,
                max_value=80,
                value=35
            )
            
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1]
            )
            
            dayofweek = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                       'Friday', 'Saturday', 'Sunday'][x]
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔮 Predict Violation Type", use_container_width=True)
        
        if predict_button:
            try:
                # Prepare input
                input_data = pd.DataFrame({
                    'Registration_State': [registration_state],
                    'Vehicle_Type': [vehicle_type],
                    'Fine_Amount': [fine_amount],
                    'Driver_Age': [driver_age],
                    'month': [month],
                    'dayofweek': [dayofweek]
                })
                
                # Encode categorical features
                for col in ['Registration_State', 'Vehicle_Type']:
                    if col in label_encoders:
                        le = label_encoders[col]
                        if input_data[col].iloc[0] in le.classes_:
                            input_data[col] = le.transform(input_data[col])
                        else:
                            input_data[col] = 0  # Unknown category
                
                # Impute missing values
                input_encoded = imputer.transform(input_data)
                
                # Predict using only Gradient Boosting model
                gb_pred = gb_model.predict(input_encoded)[0]
                gb_pred_label = le_target.inverse_transform([gb_pred])[0]
                gb_proba = gb_model.predict_proba(input_encoded)[0]
                
                # Display results
                st.success("✅ Prediction Complete!")
                st.subheader("🚀 Gradient Boosting Prediction")
                st.metric("Predicted Violation", gb_pred_label)
                st.metric("Confidence", f"{max(gb_proba)*100:.1f}%")
                
                # Show probability distribution
                st.markdown("---")
                st.subheader("📊 Prediction Probabilities (Gradient Boosting)")
                
                prob_df = pd.DataFrame({
                    'Violation Type': le_target.classes_,
                    'Probability': gb_proba
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Violation Type',
                    y='Probability',
                    title="Probability Distribution Across All Classes",
                    color='Probability',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.error("⚠️ Models not loaded. Please ensure all model files are in the directory and train the models first.")
        st.info("""
        Required files:
        - model_gradient_boosting_tuned.joblib
        - label_encoder_target.joblib
        - label_encoders.joblib
        - imputer.joblib
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚦 Traffic Violation Analytics Dashboard | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)