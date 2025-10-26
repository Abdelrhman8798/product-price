# ... (الجزء العلوي من الكود كما هو)

# ------------------------------------------------
# 3. بناء واجهة المستخدم (UI) والمدخلات
# ------------------------------------------------

# ... (الجزء الخاص بـ col_input و col_num)

        # المدخلات الرقمية (Numerical)
        with col_num:
            st.markdown("**Numerical Features**")
            
            # تم إضافة .0 لجميع قيم value و min_value
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
            
            # >>>>>>>>>> تم التعديل هنا لإزالة max_value=1.0 <<<<<<<<<<
            defect_rates = st.number_input("Defect rates", min_value=0.0, value=0.1, step=0.01, format="%.2f")

# ... (بقية الكود كما هو)
