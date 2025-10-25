import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np 
import warnings

# إخفاء تحذيرات Scikit-learn (التي تظهر بعد حل المشكلة)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------
# 1. تعريف الملفات والتحميل الآمن
# ------------------------------------------------
PIPELINE_FILE = 'simplified_price_pipeline.joblib'
FEATURES_FILE = 'final_feature_names.joblib'

@st.cache_resource
def load_assets():
    """تحميل الـ Pipeline المُبسَّط وقائمة الأعمدة النهائية."""
    if not (os.path.exists(PIPELINE_FILE) and os.path.exists(FEATURES_FILE)):
        st.error("Model files or feature list not found. Please check deployment.")
        return None, None
    try:
        pipeline = joblib.load(PIPELINE_FILE)
        final_cols = joblib.load(FEATURES_FILE)
        return pipeline, final_cols
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

pipeline, final_cols = load_assets()

# ------------------------------------------------
# 2. تعريف الفئات الثابتة (يجب أن تتطابق مع البيانات الأصلية)
# ------------------------------------------------
if pipeline is not None:
    # يجب أن تكون هذه القوائم دقيقة وتطابق القيم الفريدة في بيانات التدريب
    product_types = ['skincare', 'haircare', 'cosmetics']
    shipping_carriers = ['Carrier A', 'Carrier B', 'Carrier C']
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai'] 
    transportation_modes = ['Road', 'Rail', 'Air', 'Sea'] 


# ------------------------------------------------
# 3. بناء واجهة المستخدم (UI) والمدخلات
# ------------------------------------------------

st.set_page_config(page_title="Product Price Predictor", layout="wide")
st.title("💰 Supply Chain Price Prediction System")

if pipeline is not None:
    
    st.subheader("Enter Product and Logistics Features")

    col1, col2, col3 = st.columns(3)

    # =================================================================
    # COL 1: Product and Manufacturing Details
    # =================================================================
    with col1:
        st.header("Product & Manufacturing")
        
        product_type = st.selectbox("Product type", product_types)
        
        # استخدام st.number_input بدلاً من st.slider
        # يرجى تعديل القيم min_value, max_value, و value حسب بياناتك
        production_volumes = st.number_input("Production volumes", min_value=1.0, max_value=5000.0, value=1000.0, format="%.0f")
        manufacturing_costs = st.number_input("Manufacturing costs ($)", min_value=10.0, max_value=200.0, value=50.0, format="%.2f")
        manufacturing_lead_time = st.number_input("Manufacturing lead time (Days)", min_value=1, max_value=30, value=15, format="%.0f")
        defect_rates = st.number_input("Defect rates", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        order_quantities = st.number_input("Order quantities", min_value=1.0, max_value=2000.0, value=500.0, format="%.0f")


    # =================================================================
    # COL 2: Location and Shipping
    # =================================================================
    with col2:
        st.header("Location & Shipping")

        location = st.selectbox("Location", locations)
        shipping_carriers_input = st.selectbox("Shipping carriers", shipping_carriers)
        transportation_modes_input = st.selectbox("Transportation modes", transportation_modes)

        # استخدام st.number_input بدلاً من st.slider
        lead_time = st.number_input("Lead time (Days)", min_value=1, max_value=40, value=20, format="%.0f")
        shipping_times = st.number_input("Shipping times (Days)", min_value=1, max_value=15, value=7, format="%.0f")
        shipping_costs = st.number_input("Shipping costs ($)", min_value=1.0, max_value=30.0, value=15.0, format="%.2f")
        
        # وضع حقل Costs في Col 2
        costs = st.number_input("Costs ($)", min_value=50.0, max_value=300.0, value=150.0, format="%.2f")

    # =================================================================
    # COL 3: Prediction Button and Results
    # =================================================================
    with col3:
        st.header("Prediction")
        st.markdown("\n\n\n\n\n") # إضافة مسافة فاصلة

        if st.button("PREDICT PRICE", type="primary"):
            
            # بناء DataFrame من مدخلات المستخدم بالترتيب الأصلي
            user_data = pd.DataFrame({
                'Product type': [product_type], 
                'Production volumes': [production_volumes], 
                'Manufacturing costs': [manufacturing_costs], 
                'Manufacturing lead time': [manufacturing_lead_time],
                'Defect rates': [defect_rates], 
                'Lead time': [lead_time], 
                'Shipping times': [shipping_times],
                'Shipping carriers': [shipping_carriers_input], 
                'Shipping costs': [shipping_costs],
                'Transportation modes': [transportation_modes_input], 
                'Order quantities': [order_quantities],
                'Costs': [costs], 
                'Location': [location]
            })
            
            # 1. تحديد الأعمدة الفئوية
            categorical_cols = ['Product type', 'Shipping carriers', 'Location', 'Transportation modes']
            
            # 2. تطبيق One-Hot Encoding
            user_data_encoded = pd.get_dummies(user_data, columns=categorical_cols, drop_first=False)
            
            # 3. التأكد من أن جميع الأعمدة رقمية
            for col in user_data_encoded.columns:
                if user_data_encoded[col].dtype == bool:
                    user_data_encoded[col] = user_data_encoded[col].astype(int)

            # 4. مطابقة الأعمدة النهائية (The Crucial Step)
            final_input_data = user_data_encoded.reindex(columns=final_cols, fill_value=0)
            
            # 5. التوقع
            try:
                predicted_price = pipeline.predict(final_input_data)
                
                st.metric(
                    label="Predicted Product Price", 
                    value=f"${predicted_price[0]:.2f}", 
                    delta="Model Confidence High" # يمكنك تغيير الرسالة
                )
                st.balloons()
            except Exception as e:
                st.error(f"Prediction Error: {e}")
