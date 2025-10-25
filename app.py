import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np # قد يكون مطلوباً لمكتبة joblib

# ------------------------------------------------
# 1. تعريف الملفات والتحميل الآمن
# ------------------------------------------------
PIPELINE_FILE = 'simplified_price_pipeline.joblib'
FEATURES_FILE = 'final_feature_names.joblib'

@st.cache_resource
def load_assets():
    """تحميل الـ Pipeline المُبسَّط وقائمة الأعمدة النهائية."""
    if not (os.path.exists(PIPELINE_FILE) and os.path.exists(FEATURES_FILE)):
        st.error("ملفات الموديل أو قائمة الأعمدة غير موجودة. تأكد من رفعها جميعاً.")
        return None, None

    try:
        pipeline = joblib.load(PIPELINE_FILE)
        final_cols = joblib.load(FEATURES_FILE)
        return pipeline, final_cols
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الموديل (joblib.load): {e}")
        return None, None

pipeline, final_cols = load_assets()

# ------------------------------------------------
# 2. تعريف الفئات الثابتة (يجب أن تتطابق مع البيانات الأصلية)
# ------------------------------------------------
if pipeline is not None:
    # هذه الفئات يجب استخراجها من الـ Notebook والتأكد من أنها تغطي كل القيم الفريدة
    product_types = ['skincare', 'haircare', 'cosmetics']
    shipping_carriers = ['Carrier A', 'Carrier B', 'Carrier C']
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai'] 
    transportation_modes = ['Road', 'Rail', 'Air', 'Sea'] 
    
    # قائمة الأعمدة الأصلية التي سيتم طلبها من المستخدم
    original_features = [
        'Product type', 'Production volumes', 'Manufacturing costs',
        'Manufacturing lead time', 'Defect rates', 'Lead time',
        'Shipping times', 'Shipping carriers', 'Shipping costs',
        'Transportation modes', 'Order quantities', 'Costs', 'Location'
    ]


# ------------------------------------------------
# 3. بناء واجهة المستخدم (UI) والمدخلات
# ------------------------------------------------

st.set_page_config(page_title="توقع سعر المنتج (المعالجة المبسَّطة)", layout="wide")
st.title("💰 نظام توقع أسعار المنتجات")

if pipeline is not None:
    
    col1, col2, col3 = st.columns(3)

    # المدخلات الرقمية
    with col1:
        st.header("الكميات والتكاليف")
        # يجب تعريف هذه المتغيرات
        production_volumes = st.number_input("حجم الإنتاج (وحدة)", min_value=1.0, value=200.0, format="%.0f")
        manufacturing_costs = st.number_input("تكاليف التصنيع ($)", min_value=1.0, value=45.0, format="%.2f")
        shipping_costs = st.number_input("تكاليف الشحن ($)", min_value=0.0, value=5.0, format="%.2f")
        costs = st.number_input("التكاليف الكلية ($)", min_value=1.0, value=150.0, format="%.2f")
        order_quantities = st.number_input("كميات الطلب (وحدة)", min_value=1.0, value=100.0, format="%.0f")

    with col2:
        st.header("أوقات التسليم والعيوب")
        # يجب تعريف هذه المتغيرات
        manufacturing_lead_time = st.slider("مهلة التصنيع (يوم)", 1, 30, 15)
        lead_time = st.slider("مهلة التسليم الكلية (يوم)", 1, 30, 20)
        shipping_times = st.slider("أوقات الشحن (يوم)", 1, 15, 7)
        defect_rates = st.slider("معدل العيوب", 0.0, 1.0, 0.1, step=0.01, format="%.2f")

    # المدخلات التصنيفية
    with col3:
        st.header("الفئات والمواقع")
        # يجب تعريف هذه المتغيرات
        product_type = st.selectbox("نوع المنتج", product_types)
        shipping_carriers_input = st.selectbox("شركة الشحن", shipping_carriers)
        location = st.selectbox("الموقع", locations)
        transportation_modes_input = st.selectbox("وسيلة النقل", transportation_modes)


    # ------------------------------------------------
    # 4. معالجة المدخلات والتوقع
    # ------------------------------------------------
    if st.button("توقع السعر"):
        
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
        
        # تحديد الأعمدة الفئوية
        categorical_cols = ['Product type', 'Shipping carriers', 'Location', 'Transportation modes']
        
        # 1. تطبيق One-Hot Encoding (محاكاة لما حدث في Notebook)
        user_data_encoded = pd.get_dummies(user_data, columns=categorical_cols, drop_first=False)
        
        # 2. التأكد من أن جميع الأعمدة رقمية
        for col in user_data_encoded.columns:
            if user_data_encoded[col].dtype == bool:
                user_data_encoded[col] = user_data_encoded[col].astype(int)

        # 3. مطابقة الأعمدة النهائية (The Crucial Step)
        # نستخدم قائمة الأعمدة النهائية (final_cols) لضمان نفس عدد وترتيب الأعمدة
        final_input_data = user_data_encoded.reindex(columns=final_cols, fill_value=0)
        
        # 4. التوقع باستخدام الـ Pipeline البسيط (سيقوم بعمل Scaling ثم Prediction)
        try:
            predicted_price = pipeline.predict(final_input_data)
            st.success(f"💰 السعر المتوقع للمنتج هو: **${predicted_price[0]:.2f}**")
            st.balloons()
        except Exception as e:
            st.error(f"حدث خطأ أثناء التوقع. تأكد من توافق الإصدارات (Error: {e})")