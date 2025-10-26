import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np 
import warnings

# إخفاء تحذيرات Scikit-learn (الناتجة عن عدم تطابق الإصدارات أثناء النشر)
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
        st.error("Model assets not found. Please check deployment.")
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
st.title("💰 Product Price Prediction")
st.markdown("---")


if pipeline is not None:
    
    col_input, col_result = st.columns([2, 1])

    # =================================================================
    # COL_INPUT: جميع المدخلات في عمودين
    # =================================================================
    with col_input:
        st.subheader("1. Enter Product Details")
        
        col_cat, col_num = st.columns(2)

        # المدخلات الفئوية (Categorical)
        with col_cat:
            st.markdown("**Categorical Features**")
            product_type = st.selectbox("Product type", product_types)
            location = st.selectbox("Location", locations)
            shipping_carriers_input = st.selectbox("Shipping carriers", shipping_carriers)
            transportation_modes_input = st.selectbox("Transportation modes", transportation_modes)

        # المدخلات الرقمية (Numerical) - جميع القيم الافتراضية والحدود الدنيا تم تحويلها لـ Float (.0)
        with col_num:
            st.markdown("**Numerical Features**")
            
            # قيم الكميات والتكاليف
            # تم إضافة .0 لجميع قيم value و min_value لحل تحذير NumberInput
            production_volumes = st.number_input("Production volumes", min_value=0.0, value=1000.0, format="%.0f")
            order_quantities = st.number_input("Order quantities", min_value=0.0, value=500.0, format="%.0f")
            manufacturing_costs = st.number_input("Manufacturing costs ($)", min_value=0.0, value=50.0, format="%.2f")
            costs = st.number_input("Costs ($)", min_value=0.0, value=150.0, format="%.2f")
            shipping_costs = st.number_input("Shipping costs ($)", min_value=0.0, value=15.0, format="%.2f")
            
            # قيم زمنية
            manufacturing_lead_time = st.number_input("Manufacturing lead time (Days)", min_value=0.0, value=15.0, format="%.0f")
            lead_time = st.number_input("Lead time (Days)", min_value=0.0, value=20.0, format="%.0f")
            shipping_times = st.number_input("Shipping times (Days)", min_value=0.0, value=7.0, format="%.0f")
            
            # معدل العيوب (تم إزالة max_value=1.0)
            defect_rates = st.number_input("Defect rates", min_value=0.0, value=0.1, step=0.01, format="%.2f")

    # =================================================================
    # COL_RESULT: زر التوقع والنتائج
    # =================================================================
    with col_result:
        st.subheader("2. Prediction Result")
        st.markdown("\n\n\n\n\n\n\n")

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
                # التحقق من أن جميع الأعمدة الرقمية هي float64 أو متوافقة
                final_input_data = final_input_data.astype(np.float64)
                
                predicted_price = pipeline.predict(final_input_data)
                
                st.metric(
                    label="Predicted Product Price", 
                    value=f"${predicted_price[0]:.2f}", 
                )
            except Exception as e:
                st.error(f"Prediction Error: {e}")
