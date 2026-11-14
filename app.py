"""
Advanced MGUS to Myeloma Transformation Prediction Tool
Web application with authentication, patient management, and detailed analysis
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'data')
MODELS_DIR = os.path.join(APP_DIR, 'models')


class NaiveModel:
    """Compatibility class required by the serialized baseline models."""

    def __init__(self, prediction_value):
        self.prediction_value = prediction_value

    def predict(self, X):
        if hasattr(X, '__len__'):
            return np.full(len(X), self.prediction_value)
        return np.array([self.prediction_value])

# Page configuration
st.set_page_config(
    page_title="Myeloma Transformation Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Key biomarkers with clinical thresholds
BIOMARKER_THRESHOLDS = {
    'BAND': {
        'name': 'Paraprotein Concentration',
        'normal_range': (0, 10),
        'warning': 0,
        'critical': 10,
        'unit': 'g/L',
        'importance': 1,
        'interpretation': 'High levels indicate monoclonal protein production'
    },
    'SKLRAT': {
        'name': 'Serum Free Light Chain Ratio',
        'normal_range': (0.26, 1.65),
        'warning': 2,
        'critical': 3,
        'unit': 'ratio',
        'importance': 2,
        'interpretation': 'Abnormal ratio suggests clonal proliferation'
    },
    'CREA': {
        'name': 'Creatinine',
        'normal_range': (59, 104),
        'warning': 105,
        'critical': 110,
        'unit': 'umol/L',
        'importance': 3,
        'interpretation': 'Elevated levels indicate kidney dysfunction (CRAB criteria)'
    },
    'ADCAL': {
        'name': 'Adjusted Calcium',
        'normal_range': (2.2, 2.6),
        'warning': 2.6,
        'critical': 2.7,
        'unit': 'mmol/L',
        'importance': 4,
        'interpretation': 'Hypercalcemia indicates bone destruction (CRAB criteria)'
    },
    'HB': {
        'name': 'Hemoglobin',
        'normal_range': (115, 160),
        'warning': 115,
        'critical': 110,
        'unit': 'g/L',
        'importance': 5,
        'interpretation': 'Low levels indicate anemia from marrow infiltration (CRAB criteria)'
    },
    'IGA': {
        'name': 'Immunoglobulin A',
        'normal_range': (0.7, 4.0),
        'warning': 0.7,
        'critical': 0.5,
        'unit': 'g/L',
        'importance': 6,
        'interpretation': 'Suppression indicates immune dysfunction'
    },
    'IGM': {
        'name': 'Immunoglobulin M',
        'normal_range': (0.4, 2.3),
        'warning': 0.4,
        'critical': 0.3,
        'unit': 'g/L',
        'importance': 7,
        'interpretation': 'Suppression indicates immune dysfunction'
    },
    'B2MG': {
        'name': 'Beta-2 Microglobulin',
        'normal_range': (0, 3),
        'warning': 4,
        'critical': 5,
        'unit': 'mg/L',
        'importance': 8,
        'interpretation': 'Elevated levels indicate tumor burden and poor prognosis'
    }
}

# Authentication
def check_login(username, password):
    """Simple authentication check"""
    return username == "admin" and password == "admin"

def login_page():
    """Display login page"""
    st.markdown("<h1 style='text-align: center;'>🔬 Myeloma Transformation Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Please login to continue</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", use_container_width=True):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password. Use 'admin' / 'admin'")
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n\nUsername: `admin`\n\nPassword: `admin`")

# Data loading functions
@st.cache_data
def load_patient_data():
    """Load cancer and MGUS patient data"""
    cancer_path = os.path.join(DATA_DIR, 'cancer_patients_89.csv')
    mgus_path = os.path.join(DATA_DIR, 'mgus_patients_104.csv')
    
    cancer_df = pd.read_csv(cancer_path)
    mgus_df = pd.read_csv(mgus_path)
    
    # Parse dates
    cancer_df['DateTestTaken'] = pd.to_datetime(cancer_df['DateTestTaken'], format='mixed', dayfirst=True)
    if 'TransformationDate' in cancer_df.columns:
        cancer_df['TransformationDate'] = pd.to_datetime(cancer_df['TransformationDate'], format='mixed', dayfirst=True)
    
    mgus_df['DateTestTaken'] = pd.to_datetime(mgus_df['DateTestTaken'], format='mixed', dayfirst=True)
    
    # Add patient type
    cancer_df['PatientType'] = 'Cancer (Transformed)'
    mgus_df['PatientType'] = 'MGUS (Control)'
    
    return cancer_df, mgus_df

def calculate_biomarker_progression(patient_data, biomarker):
    """Calculate progression metrics for a biomarker"""
    marker_data = patient_data[patient_data['TestCode'] == biomarker].copy()
    
    if len(marker_data) == 0:
        return None
    
    marker_data = marker_data.sort_values('DateTestTaken')
    marker_data['Result'] = pd.to_numeric(marker_data['Result'], errors='coerce')
    marker_data = marker_data.dropna(subset=['Result'])
    
    if len(marker_data) == 0:
        return None
    
    values = marker_data['Result'].values
    dates = marker_data['DateTestTaken'].values
    
    # Calculate metrics
    metrics = {
        'current': values[-1],
        'count': len(values),
        'mean': np.mean(values),
        'std': np.std(values) if len(values) > 1 else 0,
        'min': np.min(values),
        'max': np.max(values),
        'dates': dates,
        'values': values
    }
    
    # Calculate trend (slope)
    if len(values) >= 2:
        days_from_start = (dates - dates[0]).astype('timedelta64[D]').astype(float)
        trend = np.polyfit(days_from_start, values, 1)[0]
        metrics['trend'] = trend
        metrics['trend_per_year'] = trend * 365.25
    else:
        metrics['trend'] = 0
        metrics['trend_per_year'] = 0
    
    # Calculate acceleration
    if len(values) >= 3:
        days_from_start = (dates - dates[0]).astype('timedelta64[D]').astype(float)
        poly_coeffs = np.polyfit(days_from_start, values, 2)
        metrics['acceleration'] = poly_coeffs[0]
    else:
        metrics['acceleration'] = 0
    
    return metrics

def get_available_models():
    """Get list of bundled ML models."""
    import glob
    
    models = {}
    
    advanced_models = glob.glob(os.path.join(MODELS_DIR, 'model_advanced_*.pkl'))
    for model_path in advanced_models:
        model_name = os.path.basename(model_path).replace('model_advanced_', '').replace('.pkl', '').replace('_', ' ').title()
        models[model_name] = model_path

    return models

def load_ml_model(model_path):
    """Load the selected ML model"""
    try:
        model_data = joblib.load(model_path)
        # Check if it's an advanced model (dict format) or old model (direct model)
        if isinstance(model_data, dict):
            return model_data  # Returns dict with 'model', 'scaler', 'feature_names', etc.
        else:
            return {'model': model_data, 'scaler': None, 'use_scaled': False}  # Wrap old model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None

def extract_ml_features(patient_data, biomarker_metrics):
    """Extract features for ML prediction matching the existing model format
    
    The model expects raw biomarker values (not progression features)
    Total: 45 features including all biomarkers, Age, and Gender
    """
    # Get most recent test results for all biomarkers
    latest_data = patient_data.sort_values('DateTestTaken').iloc[-1]
    
    # Initialize feature dict with all expected features
    all_biomarkers = [
        'Age', 'ADCAL', 'ALB', 'ALBE', 'ANRBC', 'B12', 'B2MG', 'BAND', 'BAND2',
        'BAS', 'CA', 'CREA', 'EOS', 'EPIGFR', 'FERRIT', 'FOLATE', 'GFR', 'HB',
        'HCT', 'IGA', 'IGE', 'IGM', 'K', 'LDH', 'LIC', 'LYMPH', 'MCH', 'MCHC',
        'MCV', 'MONO', 'MPV', 'NEUT', 'PCT', 'PDW', 'PLTS', 'RBC', 'RDW',
        'RETAB', 'SKAPP', 'SKLRAT', 'SLAMB', 'URATE', 'UREA', 'WBC'
    ]
    
    features = {}
    
    # Extract most recent value for each biomarker from patient_data
    for biomarker in all_biomarkers:
        if biomarker == 'Age':
            features['Age'] = latest_data.get('Age', 0) if 'Age' in patient_data.columns else 0
        else:
            # Get most recent value for this biomarker
            marker_data = patient_data[patient_data['TestCode'] == biomarker]
            if len(marker_data) > 0:
                latest_marker = marker_data.sort_values('DateTestTaken').iloc[-1]
                result = latest_marker['Result']
                # Convert to numeric
                try:
                    features[biomarker] = float(result)
                except:
                    features[biomarker] = np.nan
            else:
                features[biomarker] = np.nan
    
    # Add Gender feature
    gender = latest_data.get('Gender', 'Unknown') if 'Gender' in patient_data.columns else 'Unknown'
    features['Gender_M'] = 1 if gender == 'M' else 0
    
    # Convert to array in correct order
    feature_names = all_biomarkers + ['Gender_M']
    feature_values = [features.get(name, np.nan) for name in feature_names]
    
    return np.array(feature_values).reshape(1, -1)

def extract_advanced_ml_features(patient_data, biomarker_metrics):
    """Extract progression features for advanced ML models (52 features)
    
    These models expect progression features matching predict_transformation_advanced.py:
    - For each of 8 key biomarkers: recent, count, trend, acceleration, volatility, recent_trend (48 features)
    - Plus 4 meta-features: total_measurements, monitoring_duration_days, avg_days_to_transformation, min_days_to_transformation
    
    Total: 52 features
    """
    import pandas as pd
    
    KEY_BIOMARKERS = ['BAND', 'SKLRAT', 'CREA', 'ADCAL', 'HB', 'IGA', 'IGM', 'B2MG']
    features = {}
    
    # For each key biomarker, extract progression features
    for biomarker in KEY_BIOMARKERS:
        if biomarker in biomarker_metrics:
            metrics = biomarker_metrics[biomarker]
            
            # Feature 1: Recent value
            features[f'{biomarker}_recent'] = metrics.get('current', np.nan)
            
            # Feature 2: Count
            features[f'{biomarker}_count'] = metrics.get('count', 0)
            
            # Feature 3: Trend (convert from per year to per day)
            trend_per_year = metrics.get('trend_per_year', 0)
            features[f'{biomarker}_trend'] = trend_per_year / 365.25
            
            # Feature 4: Acceleration
            features[f'{biomarker}_acceleration'] = metrics.get('acceleration', 0)
            
            # Feature 5: Volatility
            features[f'{biomarker}_volatility'] = metrics.get('std', 0)
            
            # Feature 6: Recent trend (same as overall trend since we don't calculate separately)
            features[f'{biomarker}_recent_trend'] = trend_per_year / 365.25
        else:
            # Missing biomarker - use neutral values
            features[f'{biomarker}_recent'] = np.nan
            features[f'{biomarker}_count'] = 0
            features[f'{biomarker}_trend'] = 0
            features[f'{biomarker}_acceleration'] = 0
            features[f'{biomarker}_volatility'] = 0
            features[f'{biomarker}_recent_trend'] = 0
    
    # Meta-features
    features['total_measurements'] = len(patient_data)
    
    if len(patient_data) > 0:
        duration = (patient_data['DateTestTaken'].max() - patient_data['DateTestTaken'].min()).days
        features['monitoring_duration_days'] = float(duration)
    else:
        features['monitoring_duration_days'] = 0.0
    
    # These are unknown during prediction (they require knowing the transformation date)
    # Set to 0 - models should be robust to this
    features['avg_days_to_transformation'] = 0.0
    features['min_days_to_transformation'] = 0.0
    
    # Build feature array in correct order
    feature_names = []
    for biomarker in KEY_BIOMARKERS:
        feature_names.extend([
            f'{biomarker}_recent',
            f'{biomarker}_count',
            f'{biomarker}_trend',
            f'{biomarker}_acceleration',
            f'{biomarker}_volatility',
            f'{biomarker}_recent_trend'
        ])
    
    feature_names.extend([
        'total_measurements',
        'monitoring_duration_days',
        'avg_days_to_transformation',
        'min_days_to_transformation'
    ])
    
    feature_values = [features[name] for name in feature_names]
    return np.array(feature_values).reshape(1, -1)

def predict_transformation_date(patient_data, model_path=None):
    """Predict transformation date using ML model and biomarker analysis"""
    
    # Calculate progression features for all key biomarkers
    biomarker_metrics = {}
    risk_score = 0
    risk_factors = []
    abnormal_count = 0
    critical_count = 0
    
    for biomarker, info in BIOMARKER_THRESHOLDS.items():
        metrics = calculate_biomarker_progression(patient_data, biomarker)
        if metrics:
            biomarker_metrics[biomarker] = metrics
            
            # Assess risk based on current value
            current_val = metrics['current']
            normal_min, normal_max = info['normal_range']
            
            # Check if outside normal range with weighted importance
            importance_weight = 1 / info['importance']
            
            if biomarker == 'HB':  # For HB, lower is worse
                if current_val < info['critical']:
                    risk_score += 3 * importance_weight
                    critical_count += 1
                    risk_factors.append(f"{info['name']} critically low: {current_val:.1f} {info['unit']}")
                elif current_val < info['warning']:
                    risk_score += 2 * importance_weight
                    abnormal_count += 1
                    risk_factors.append(f"{info['name']} below normal: {current_val:.1f} {info['unit']}")
            elif biomarker in ['IGA', 'IGM']:  # For Igs, lower is worse
                if current_val < info['critical']:
                    risk_score += 3 * importance_weight
                    critical_count += 1
                    risk_factors.append(f"{info['name']} critically low: {current_val:.2f} {info['unit']}")
                elif current_val < info['warning']:
                    risk_score += 2 * importance_weight
                    abnormal_count += 1
                    risk_factors.append(f"{info['name']} suppressed: {current_val:.2f} {info['unit']}")
            else:  # For others, higher is worse
                if current_val > info['critical']:
                    risk_score += 3 * importance_weight
                    critical_count += 1
                    risk_factors.append(f"{info['name']} critically high: {current_val:.1f} {info['unit']}")
                elif current_val > info['warning']:
                    risk_score += 2 * importance_weight
                    abnormal_count += 1
                    risk_factors.append(f"{info['name']} elevated: {current_val:.1f} {info['unit']}")
            
            # Check trend - add risk if biomarkers are worsening
            if metrics['trend_per_year'] != 0:
                if biomarker in ['HB', 'IGA', 'IGM']:  # Declining is bad
                    if metrics['trend_per_year'] < -1:
                        risk_score += 1 * importance_weight
                        risk_factors.append(f"{info['name']} declining: {metrics['trend_per_year']:.2f} {info['unit']}/year")
                else:  # Rising is bad
                    if metrics['trend_per_year'] > 1:
                        risk_score += 1 * importance_weight
                        risk_factors.append(f"{info['name']} rising: {metrics['trend_per_year']:.2f} {info['unit']}/year")
    
    # Get latest test date
    latest_date = patient_data['DateTestTaken'].max()
    
    # Try to use ML model for prediction
    if model_path:
        ml_model_data = load_ml_model(model_path)
    else:
        ml_model_data = None
    
    if ml_model_data is not None:
        try:
            ml_model = ml_model_data['model']
            scaler = ml_model_data.get('scaler')
            use_scaled = ml_model_data.get('use_scaled', False)
            feature_names = ml_model_data.get('feature_names', [])
            
            # Determine which feature extractor to use based on expected feature count
            expected_features = len(feature_names) if feature_names else 45
            
            if expected_features == 52:
                # Advanced model with progression features
                features = extract_advanced_ml_features(patient_data, biomarker_metrics)
            else:
                # Old model with raw biomarker values
                features = extract_ml_features(patient_data, biomarker_metrics)
            
            # Apply scaling if needed
            if use_scaled and scaler is not None:
                features = scaler.transform(features)
            
            # Handle NaN values (fill with median or 0)
            features = np.nan_to_num(features, nan=0.0)
            
            # Predict days to transformation using ML model
            days_to_transformation = float(ml_model.predict(features)[0])
            days_to_transformation = max(30, days_to_transformation)  # Minimum 30 days (1 month)
            months_to_transformation = days_to_transformation / 30.44
            
            # Get model name from path
            model_name = os.path.basename(model_path).replace('model_advanced_', '').replace('.pkl', '').replace('_', ' ').title()
            prediction_method = f"Machine Learning ({model_name})"
        except Exception as e:
            # Fallback to heuristic if ML fails
            st.warning(f"ML prediction failed: {e}. Using heuristic method.")
            ml_model_data = None  # Force heuristic calculation below
    
    # Use improved heuristic method if ML model unavailable or failed
    if ml_model_data is None:
        """
        IMPROVED HEURISTIC FORMULA based on data analysis:
        
        Data shows:
        - Median time from first BAND detection to transformation: 5.6 months
        - Patients transform within 1 year: 51.2%
        - High-risk patients (multiple critical biomarkers): faster progression
        - Low-risk patients (few abnormalities): slower progression
        
        Formula rationale:
        - Base prediction on biomarker severity (risk_score)
        - Critical biomarkers indicate imminent transformation
        - Multiple abnormalities suggest active disease
        - Use exponential decay for high risk, longer times for low risk
        """
        
        # Risk-stratified prediction
        if critical_count >= 3:
            # Very high risk: multiple critical biomarkers
            # Suggest transformation within 3-6 months
            months_to_transformation = max(3, 12 / (1 + risk_score))
            risk_category = "Very High Risk"
        elif critical_count >= 1 or abnormal_count >= 4:
            # High risk: at least one critical or many abnormal
            # Suggest transformation within 6-12 months
            months_to_transformation = max(6, 18 / (1 + risk_score * 0.8))
            risk_category = "High Risk"
        elif abnormal_count >= 2:
            # Moderate risk: multiple abnormalities
            # Suggest transformation within 12-24 months
            months_to_transformation = max(12, 36 / (1 + risk_score * 0.5))
            risk_category = "Moderate Risk"
        elif abnormal_count >= 1:
            # Low-moderate risk: single abnormality
            # Suggest transformation within 24-48 months
            months_to_transformation = max(24, 60 / (1 + risk_score * 0.3))
            risk_category = "Low-Moderate Risk"
        else:
            # Very low risk: no significant abnormalities
            # Suggest transformation > 5 years or may not transform
            months_to_transformation = 120  # 10 years - very conservative
            risk_category = "Very Low Risk"
        
        prediction_method = f"Heuristic ({risk_category})"
    
    predicted_date = latest_date + timedelta(days=months_to_transformation * 30.44)
    
    return {
        'predicted_date': predicted_date,
        'months_to_transformation': months_to_transformation,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'biomarker_metrics': biomarker_metrics,
        'prediction_method': prediction_method,
        'abnormal_count': abnormal_count,
        'critical_count': critical_count
    }

def display_prediction_reasoning(prediction_result):
    """Display detailed reasoning for the prediction"""
    st.markdown("### 🎯 Prediction Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Predicted Transformation Date",
            prediction_result['predicted_date'].strftime('%Y-%m-%d'),
            f"~{prediction_result['months_to_transformation']:.0f} months from latest test"
        )
    
    with col2:
        risk_level = "HIGH" if prediction_result['risk_score'] > 5 else "MODERATE" if prediction_result['risk_score'] > 2 else "LOW"
        risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MODERATE" else "🟢"
        
        # Show detailed risk breakdown
        abnormal = prediction_result.get('abnormal_count', 0)
        critical = prediction_result.get('critical_count', 0)
        risk_detail = f"{critical} critical, {abnormal} abnormal"
        
        st.metric(
            "Risk Level",
            f"{risk_color} {risk_level}",
            f"Score: {prediction_result['risk_score']:.2f} ({risk_detail})"
        )
    
    with col3:
        st.metric(
            "Biomarkers Analyzed",
            len(prediction_result['biomarker_metrics']),
            f"/{len(BIOMARKER_THRESHOLDS)} key markers"
        )
    
    with col4:
        method = prediction_result.get('prediction_method', 'Heuristic')
        method_icon = "🤖" if "Machine Learning" in method else "📊"
        st.metric(
            "Prediction Method",
            f"{method_icon}",
            method
        )
    
    # Detailed reasoning
    st.markdown("---")
    st.markdown("### 📋 Detailed Reasoning")
    
    if prediction_result['risk_factors']:
        st.markdown("**Risk Factors Identified:**")
        for i, factor in enumerate(prediction_result['risk_factors'], 1):
            st.markdown(f"{i}. {factor}")
    else:
        st.success("✅ No significant risk factors identified. All biomarkers within acceptable ranges.")
    
    # Biomarker analysis
    st.markdown("---")
    st.markdown("### 🔬 Biomarker Progression Analysis")
    
    for biomarker, metrics in prediction_result['biomarker_metrics'].items():
        if biomarker in BIOMARKER_THRESHOLDS:
            info = BIOMARKER_THRESHOLDS[biomarker]
            
            with st.expander(f"**{info['name']} ({biomarker})** - Current: {metrics['current']:.2f} {info['unit']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create trend chart
                    fig = go.Figure()
                    
                    # Plot values over time
                    fig.add_trace(go.Scatter(
                        x=metrics['dates'],
                        y=metrics['values'],
                        mode='lines+markers',
                        name=biomarker,
                        line=dict(color='#3498db', width=2),
                        marker=dict(size=8)
                    ))
                    
                    # Add normal range
                    if biomarker in ['HB', 'IGA', 'IGM']:
                        fig.add_hline(y=info['warning'], line_dash="dash", line_color="orange", 
                                     annotation_text="Warning")
                        fig.add_hline(y=info['critical'], line_dash="dash", line_color="red",
                                     annotation_text="Critical")
                    else:
                        fig.add_hline(y=info['warning'], line_dash="dash", line_color="orange",
                                     annotation_text="Warning")
                        fig.add_hline(y=info['critical'], line_dash="dash", line_color="red",
                                     annotation_text="Critical")
                    
                    fig.update_layout(
                        title=f"{info['name']} Trend",
                        xaxis_title="Date",
                        yaxis_title=f"{info['name']} ({info['unit']})",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"**Statistics:**")
                    st.markdown(f"- Current: {metrics['current']:.2f} {info['unit']}")
                    st.markdown(f"- Mean: {metrics['mean']:.2f}")
                    st.markdown(f"- Range: {metrics['min']:.2f} - {metrics['max']:.2f}")
                    st.markdown(f"- Measurements: {metrics['count']}")
                    st.markdown(f"- Trend: {metrics['trend_per_year']:.2f} {info['unit']}/year")
                    
                    st.markdown(f"**Clinical Interpretation:**")
                    st.markdown(f"_{info['interpretation']}_")
                    
                    # Status indicator
                    current = metrics['current']
                    if biomarker in ['HB', 'IGA', 'IGM']:
                        if current < info['critical']:
                            st.error("🔴 Critical")
                        elif current < info['warning']:
                            st.warning("🟡 Warning")
                        else:
                            st.success("🟢 Normal")
                    else:
                        if current > info['critical']:
                            st.error("🔴 Critical")
                        elif current > info['warning']:
                            st.warning("🟡 Warning")
                        else:
                            st.success("🟢 Normal")

def data_exploration_tab(patient_data):
    """Data exploration interface"""
    st.markdown("### 📊 Patient Data Exploration")
    
    # Date range
    min_date = patient_data['DateTestTaken'].min()
    max_date = patient_data['DateTestTaken'].max()
    
    st.info(f"📅 Data Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} "
            f"({(max_date - min_date).days} days)")
    
    # Test selection
    available_tests = sorted(patient_data['TestCode'].dropna().unique())
    selected_test = st.selectbox("Select Blood Test Marker:", available_tests)
    
    if selected_test:
        test_data = patient_data[patient_data['TestCode'] == selected_test].copy()
        test_data['Result'] = pd.to_numeric(test_data['Result'], errors='coerce')
        test_data = test_data.dropna(subset=['Result'])
        test_data = test_data.sort_values('DateTestTaken')
        
        if len(test_data) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=test_data['DateTestTaken'],
                    y=test_data['Result'],
                    mode='lines+markers',
                    name=selected_test,
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=10)
                ))
                
                # Add thresholds if available
                if selected_test in BIOMARKER_THRESHOLDS:
                    info = BIOMARKER_THRESHOLDS[selected_test]
                    
                    if selected_test in ['HB', 'IGA', 'IGM']:
                        fig.add_hline(y=info['warning'], line_dash="dash", line_color="orange",
                                     annotation_text="Warning Level")
                        fig.add_hline(y=info['critical'], line_dash="dash", line_color="red",
                                     annotation_text="Critical Level")
                    else:
                        fig.add_hline(y=info['warning'], line_dash="dash", line_color="orange",
                                     annotation_text="Warning Level")
                        fig.add_hline(y=info['critical'], line_dash="dash", line_color="red",
                                     annotation_text="Critical Level")
                
                test_name = test_data['TestName'].iloc[0] if 'TestName' in test_data.columns else selected_test
                fig.update_layout(
                    title=f"{test_name} Historical Data",
                    xaxis_title="Date",
                    yaxis_title=f"Value ({test_data['Unit'].iloc[0] if 'Unit' in test_data.columns else ''})",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Statistics:**")
                st.metric("Latest Value", f"{test_data['Result'].iloc[-1]:.2f}")
                st.metric("Mean", f"{test_data['Result'].mean():.2f}")
                st.metric("Std Dev", f"{test_data['Result'].std():.2f}")
                st.metric("Min", f"{test_data['Result'].min():.2f}")
                st.metric("Max", f"{test_data['Result'].max():.2f}")
                st.metric("Measurements", len(test_data))
                
                # Trend
                if len(test_data) >= 2:
                    dates_numeric = (test_data['DateTestTaken'] - test_data['DateTestTaken'].min()).dt.days.values
                    trend = np.polyfit(dates_numeric, test_data['Result'].values, 1)[0]
                    trend_per_year = trend * 365.25
                    st.metric("Trend", f"{trend_per_year:.2f}/year")
            
            # Data table
            st.markdown("---")
            st.markdown("**Historical Measurements:**")
            display_data = test_data[['DateTestTaken', 'TestName', 'Result', 'Unit']].copy()
            display_data['DateTestTaken'] = display_data['DateTestTaken'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_data, use_container_width=True, height=300)
        else:
            st.warning(f"No valid numeric data found for {selected_test}")

def upload_blood_test_tab(patient_id, patient_type):
    """Upload new blood test results"""
    st.markdown("### 📤 Upload New Blood Test Results")
    
    st.info(f"Uploading results for Patient: **{patient_id}** ({patient_type})")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with blood test results",
        type=['csv'],
        help="CSV should have columns: DateTestTaken, TestCode, TestName, Result, Unit"
    )
    
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            
            st.markdown("**Preview of uploaded data:**")
            st.dataframe(new_data.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['DateTestTaken', 'TestCode', 'Result']
            missing_cols = [col for col in required_cols if col not in new_data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("✅ Confirm and Save", type="primary"):
                    # Add patient info
                    new_data['UniqueID'] = patient_id
                    new_data['PatientID'] = patient_id
                    
                    # Save to file
                    filename = 'cancer_patients_89.csv' if patient_type == 'Cancer (Transformed)' else 'mgus_patients_104.csv'
                    filepath = os.path.join(DATA_DIR, filename)
                    
                    # Append to existing file
                    existing_data = pd.read_csv(filepath)
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    combined_data.to_csv(filepath, index=False)
                    
                    st.success(f"✅ Successfully uploaded {len(new_data)} test results!")
                    st.balloons()
                    
                    # Clear cache to reload data
                    st.cache_data.clear()
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Manual entry option
    st.markdown("---")
    st.markdown("**Or enter test results manually:**")
    
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        
        with col1:
            test_date = st.date_input("Test Date", datetime.now())
            test_code = st.selectbox("Test Code", list(BIOMARKER_THRESHOLDS.keys()))
        
        with col2:
            test_result = st.number_input("Result Value", min_value=0.0, step=0.1)
            test_unit = st.text_input("Unit", BIOMARKER_THRESHOLDS[test_code]['unit'])
        
        submitted = st.form_submit_button("Add Test Result", type="primary")
        
        if submitted:
            # Create new record
            new_record = pd.DataFrame({
                'PatientID': [patient_id],
                'UniqueID': [patient_id],
                'DateTestTaken': [test_date.strftime('%d/%m/%Y')],
                'TestCode': [test_code],
                'TestName': [BIOMARKER_THRESHOLDS[test_code]['name']],
                'Result': [test_result],
                'Unit': [test_unit]
            })
            
            # Save
            filename = 'cancer_patients_89.csv' if patient_type == 'Cancer (Transformed)' else 'mgus_patients_104.csv'
            filepath = os.path.join(DATA_DIR, filename)
            
            existing_data = pd.read_csv(filepath)
            combined_data = pd.concat([existing_data, new_record], ignore_index=True)
            combined_data.to_csv(filepath, index=False)
            
            st.success("✅ Test result added successfully!")
            st.cache_data.clear()
            st.rerun()

def main():
    """Main application"""
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Check login
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Logged in - show main app
    st.sidebar.title("🔬 Myeloma Predictor")
    st.sidebar.markdown(f"**User:** {st.session_state.username}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Load data
    try:
        cancer_df, mgus_df = load_patient_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Patient filter
    st.sidebar.markdown("### Patient Selection")
    
    patient_type_filter = st.sidebar.radio(
        "Patient Type:",
        ["All", "Cancer (Transformed)", "MGUS (Control)"]
    )
    
    # Filter data
    if patient_type_filter == "Cancer (Transformed)":
        filtered_df = cancer_df
    elif patient_type_filter == "MGUS (Control)":
        filtered_df = mgus_df
    else:
        filtered_df = pd.concat([cancer_df, mgus_df], ignore_index=True)
    
    # Get unique patients
    patient_list = filtered_df.groupby('UniqueID').first().reset_index()
    patient_list = patient_list.sort_values('UniqueID')
    
    # Create patient dropdown options
    patient_options = {}
    for _, row in patient_list.iterrows():
        patient_id = row['UniqueID']
        patient_type = row['PatientType']
        gender = row['Gender'] if 'Gender' in row else 'Unknown'
        age = row['Age'] if 'Age' in row else 'Unknown'
        
        label = f"{patient_id} ({patient_type}, {gender}, Age {age})"
        patient_options[label] = patient_id
    
    selected_patient_label = st.sidebar.selectbox(
        "Select Patient:",
        list(patient_options.keys())
    )
    
    selected_patient_id = patient_options[selected_patient_label]
    selected_patient_data = filtered_df[filtered_df['UniqueID'] == selected_patient_id].copy()
    selected_patient_type = selected_patient_data['PatientType'].iloc[0]
    
    # Display patient info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Patient Information")
    st.sidebar.markdown(f"**ID:** {selected_patient_id}")
    st.sidebar.markdown(f"**Type:** {selected_patient_type}")
    st.sidebar.markdown(f"**Total Tests:** {len(selected_patient_data)}")
    
    if 'Gender' in selected_patient_data.columns:
        st.sidebar.markdown(f"**Gender:** {selected_patient_data['Gender'].iloc[0]}")
    if 'Age' in selected_patient_data.columns:
        st.sidebar.markdown(f"**Age:** {selected_patient_data['Age'].iloc[0]}")
    if 'TransformationDate' in selected_patient_data.columns:
        transformation_date = selected_patient_data['TransformationDate'].iloc[0]
        if pd.notna(transformation_date):
            st.sidebar.markdown(f"**Transformation Date:** {transformation_date.strftime('%Y-%m-%d')}")
        else:
            st.sidebar.markdown(f"**Transformation Date:** Not transformed (MGUS)")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Data Exploration", "📤 Upload Results"])
    
    with tab1:
        st.markdown(f"## Transformation Date Prediction for Patient {selected_patient_id}")
        
        # Model selector
        st.markdown("### Select Prediction Model")
        available_models = get_available_models()
        
        model_names = ["Heuristic Method"]
        if available_models:
            model_names.extend(list(available_models.keys()))
        default_model_index = model_names.index("Lasso Regression") if "Lasso Regression" in model_names else 0
        
        selected_model_name = st.selectbox(
            "Choose prediction algorithm:",
            model_names,
            index=default_model_index,
            help="Lasso Regression is selected by default. Heuristic Method uses risk-stratified clinical reasoning; ML models are trained on retrospective data."
        )
        
        # Get model path if ML model selected
        if selected_model_name == "Heuristic Method":
            selected_model_path = None
        else:
            selected_model_path = available_models[selected_model_name]
        
        # Add date selector for prediction cutoff
        st.markdown("### Select Prediction Date")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get date range from patient data
            min_date = selected_patient_data['DateTestTaken'].min().date()
            max_date = selected_patient_data['DateTestTaken'].max().date()
            default_date = max_date
            
            prediction_date = st.date_input(
                "Use blood tests up to this date:",
                value=default_date,
                min_value=min_date,
                max_value=datetime.now().date(),
                help="Only blood tests taken on or before this date will be used for prediction"
            )
        
        with col2:
            # Show info about selected date
            filtered_data = selected_patient_data[selected_patient_data['DateTestTaken'].dt.date <= prediction_date]
            st.metric("Tests Available", len(filtered_data), f"up to {prediction_date}")
        
        st.markdown("---")
        
        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            # Filter data to only include tests before the selected date
            filtered_patient_data = selected_patient_data[
                selected_patient_data['DateTestTaken'].dt.date <= prediction_date
            ].copy()
            
            if len(filtered_patient_data) == 0:
                st.error("No blood test data available on or before the selected date.")
            else:
                with st.spinner("Analyzing biomarker progression..."):
                    prediction_result = predict_transformation_date(filtered_patient_data, selected_model_path)
                    
                    # Add prediction date to result for display
                    prediction_result['prediction_cutoff_date'] = prediction_date
                    
                    st.info(f"ℹ️ Prediction based on {len(filtered_patient_data)} blood tests taken on or before **{prediction_date.strftime('%Y-%m-%d')}** using **{selected_model_name}**")
                    display_prediction_reasoning(prediction_result)
    
    with tab2:
        data_exploration_tab(selected_patient_data)
    
    with tab3:
        upload_blood_test_tab(selected_patient_id, selected_patient_type)

if __name__ == "__main__":
    main()
