# ... (الجزء العلوي من الكود كما هو)

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

        # المدخلات الرقمية (Numerical) - تم التعديل إلى Float (.0) لإزالة التحذير
        with col_num:
            st.markdown("**Numerical Features**")
            
            # تم إضافة .0 لجميع قيم value و min_value (إذا كانت الأرقام صحيحة)
            production_volumes = st.number_input("Production volumes", min_value=0.0, value=1000.0, format="%.0f")
            manufacturing_costs = st.number_input("Manufacturing costs ($)", min_value=0.0, value=50.0, format="%.2f")
            costs = st.number_input("Costs ($)", min_value=0.0, value=150.0, format="%.2f")
            order_quantities = st.number_input("Order quantities", min_value=0.0, value=500.0, format="%.0f")
            
            # قيم زمنية
            manufacturing_lead_time = st.number_input("Manufacturing lead time (Days)", min_value=0.0, value=15.0, format="%.0f")
            lead_time = st.number_input("Lead time (Days)", min_value=0.0, value=20.0, format="%.0f")
            shipping_times = st.number_input("Shipping times (Days)", min_value=0.0, value=7.0, format="%.0f")
            
            # قيم التكاليف والنسب
            shipping_costs = st.number_input("Shipping costs ($)", min_value=0.0, value=15.0, format="%.2f")
            defect_rates = st.number_input("Defect rates", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")

    # ... (بقية الكود الخاص بالتوقع كما هو)
    
    with col_result:
        st.subheader("2. Prediction Result")
        st.markdown("\n\n\n\n\n\n\n")

        if st.button("PREDICT PRICE", type="primary"):
            # ... (بقية منطق التوقع كما هو)
            # يجب التأكد من استخدام متغيرات الإدخال الجديدة في DataFrame
            
            # ... (منطق التوقع)
            try:
                predicted_price = pipeline.predict(final_input_data)
                
                st.metric(
                    label="Predicted Product Price", 
                    value=f"${predicted_price[0]:.2f}", 
                )
            except Exception as e:
                st.error(f"Prediction Error: {e}")
