import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import streamlit.components.v1 as components
import io
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
    }
    
    .prediction-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(81, 207, 102, 0.3);
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        model = load_model('model.h5')
        
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('ohe_encoder_gender.pkl', 'rb') as file:
            ohe_encoder_gender = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return model, label_encoder_gender, ohe_encoder_gender, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

model, label_encoder_gender, ohe_encoder_gender, scaler = load_models()

# Header
st.markdown('<h1 class="main-header">🎯 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced ML-powered customer retention analytics</p>', unsafe_allow_html=True)

# Sidebar with app info
with st.sidebar:
    st.markdown("### 📋 About This App")
    st.markdown("""
    This application uses advanced machine learning to predict customer churn probability.
    
    **Features:**
    - 🤖 Deep Learning Model
    - 📊 Real-time Predictions
    - 📈 Interactive Visualizations
    - 📋 Downloadable Reports
    """)
    
    st.markdown("### 🎨 Prediction Legend")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #51cf66 0%, #40c057 100%); padding: 10px; border-radius: 5px; margin: 5px 0;">
        <span style="color: white;">🟢 Low Risk (< 50%)</span>
    </div>
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 10px; border-radius: 5px; margin: 5px 0;">
        <span style="color: white;">🔴 High Risk (≥ 50%)</span>
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.error("Failed to load models. Please check if all model files are available.")
    st.stop()

# Input Section
st.markdown("## 📝 Customer Information")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "💰 Financial Info", "📊 Account Details"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox('🌍 Geography', ohe_encoder_gender.categories_[0])
        gender = st.selectbox('👥 Gender', label_encoder_gender.classes_)
    with col2:
        age = st.slider('🎂 Age', 18, 92, 35)
        tenure = st.slider('📅 Tenure (Years)', 0, 10, 5)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
        balance = st.number_input('💰 Account Balance ($)', min_value=0.0, value=50000.0, step=1000.0)
    with col2:
        estimated_salary = st.number_input('💼 Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0)

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        num_of_products = st.slider('📦 Number of Products', 1, 4, 2)
    with col2:
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    with col3:
        is_active_member = st.selectbox('✅ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Prediction button
if st.button("🔮 Predict Churn Probability", type="primary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    
    # One-hot encode geography
    geo_encoded = ohe_encoder_gender.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_encoder_gender.get_feature_names_out(['Geography']))
    
    # Combine data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Scale the data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    predic_prob = prediction[0][0]
    
    # Display results
    st.markdown("## 🎯 Prediction Results")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = predic_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction message
    if predic_prob > 0.5:
        st.markdown(f"""
        <div class="prediction-high">
            <h2>⚠️ HIGH CHURN RISK</h2>
            <h3>Probability: {predic_prob*100:.2f}%</h3>
            <p>This customer is likely to churn. Consider implementing retention strategies!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        recommendations = [
            "🎯 Offer personalized retention incentives",
            "📞 Schedule a customer success call",
            "💰 Provide special pricing or discounts",
            "🎁 Offer additional services or products",
            "📧 Send targeted email campaigns"
        ]
        for rec in recommendations:
            st.markdown(f"- {rec}")
            
    else:
        st.markdown(f"""
        <div class="prediction-low">
            <h2>✅ LOW CHURN RISK</h2>
            <h3>Probability: {predic_prob*100:.2f}%</h3>
            <p>This customer is likely to stay. Great job on customer satisfaction!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommended Actions")
        recommendations = [
            "🌟 Maintain current service quality",
            "📈 Explore upselling opportunities",
            "🤝 Build stronger relationship",
            "📋 Request feedback and testimonials",
            "🎉 Recognize customer loyalty"
        ]
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    # Feature importance visualization
    st.markdown("### 📊 Customer Profile Summary")
    
    # Create a feature summary
    features_df = pd.DataFrame({
        'Feature': ['Credit Score', 'Age', 'Balance', 'Estimated Salary', 'Tenure', 'Num Products'],
        'Value': [credit_score, age, balance, estimated_salary, tenure, num_of_products],
        'Category': ['Financial', 'Demographic', 'Financial', 'Financial', 'Behavioral', 'Behavioral']
    })
    
    fig_bar = px.bar(features_df, x='Feature', y='Value', color='Category',
                     title="Customer Profile Overview",
                     color_discrete_map={'Financial': '#667eea', 'Demographic': '#764ba2', 'Behavioral': '#f093fb'})
    fig_bar.update_layout(showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Download report
    st.markdown("### 📥 Download Report")
    
    # Create detailed report
    report_data = input_data.copy()
    report_data["Churn_Probability"] = predic_prob
    report_data["Risk_Level"] = "High" if predic_prob > 0.5 else "Low"
    report_data["Prediction_Date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert to CSV
    output = io.StringIO()
    report_data.to_csv(output, index=False)
    
    st.download_button(
        label="📋 Download Detailed Report",
        data=output.getvalue(),
        file_name=f"churn_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download a detailed CSV report of the prediction"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 Powered by Advanced Machine Learning | Built with Streamlit</p>
    <p>For support or questions, contact your data science team</p>
</div>
""", unsafe_allow_html=True)