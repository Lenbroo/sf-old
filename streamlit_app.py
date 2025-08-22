import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
# Assuming style_utils.py exists and contains the necessary functions
from style_utils import load_css, load_footer
import streamlit.components.v1 as components

# Initialize session state for DataFrames
if 'output_df' not in st.session_state:
    st.session_state.output_df = None
if 'pivot_df' not in st.session_state:
    st.session_state.pivot_df = None
if 'unique_patients_pivot' not in st.session_state:
    st.session_state.unique_patients_pivot = None

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Smart Flow", layout="centered")

# Load shared CSS and footer
load_css()

components.html(
    """
    <script>
      const meta = document.createElement('meta');
      meta.name = "color-scheme";
      meta.content = "light dark";
      document.head.appendChild(meta);
    </script>
    """,
    height=0,
)
st.title("🧠 Smart Flow")
st.markdown("""
Drop your Excel report below and let the magic begin. ✨.
""")

# ---------- File uploaders ----------
uploaded_file = st.file_uploader("Upload Main Excel File", type=["xlsx"])
uploaded_payment_file = st.file_uploader("Upload Payment Excel File (Optional)", type=["xlsx"])

def format_date_to_string(date_value, date_value_end=None):
    """Convert date(s) to string in '24th of July 2025' format or 'for the period x to z'."""
    def format_single_date(date_val):
        if pd.isna(date_val):
            return ""
        try:
            if isinstance(date_val, str):
                date_obj = pd.to_datetime(date_val)
            else:
                date_obj = date_val
            day = date_obj.day
            if 10 <= day % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
            return f"{day}{suffix} of {date_obj.strftime('%B %Y')}"
        except (ValueError, TypeError):
            return str(date_val)

    if date_value_end is None or date_value == date_value_end:
        return format_single_date(date_value)
    else:
        start_date = format_single_date(date_value)
        end_date = format_single_date(date_value_end)
        return f"for the period {start_date} to {end_date}"

if uploaded_file:
    try:
        # Stage 1: Basic filtering and formatting
        df = pd.read_excel(uploaded_file)
        st.session_state.raw_df = df.copy()
        df = df[df['packageprice'].notna() & (df['packageprice'] != 0)].reset_index(drop=True)

        output_df = pd.DataFrame()
        output_df['No'] = range(1, len(df) + 1)
        output_df['Date'] = df['billdate'].apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y') if pd.notnull(x) else '')
        output_df['Name'] = df['patientname']
        output_df['Location'] = df['locationcenter']
        output_df['Package'] = df['servicename']
        output_df['Payment'] = ''
        output_df['Time'] = df['billtime']
        output_df['Shopify'] = df['transactionno']
        output_df['Desc Code'] = df['discountcode']
        output_df['Price'] = df['netamt']
        output_df['Hrs'] = ''
        output_df['Under Process'] = df['availeddatetime'].apply(lambda x: 'UNDER PROCESS' if pd.notnull(x) else 'SAMPLE PENDING')
        output_df.loc[output_df['Price'] == 0, 'Payment'] = '-'

        # Get min and max dates for the header
        valid_dates = pd.to_datetime(df['billdate'].dropna(), errors='coerce')
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            header_date = format_date_to_string(min_date, max_date)
        else:
            header_date = format_date_to_string(datetime.today())

        # Process payment file if uploaded
        if uploaded_payment_file:
            try:
                payment_df = pd.read_excel(uploaded_payment_file, skiprows=9, header=None)

                if payment_df.shape[1] < 20:
                    st.warning("⚠️ The uploaded payment file does not have enough columns to be processed. Proceeding without payment data.")
                else:
                    payment_df = payment_df.iloc[:, [5, 17, 18, 19]]
                    payment_df = payment_df.dropna(subset=[5]).reset_index(drop=True)
                    payment_df[17] = pd.to_numeric(payment_df[17], errors='coerce')
                    payment_df[18] = pd.to_numeric(payment_df[18], errors='coerce')
                    payment_df[19] = pd.to_numeric(payment_df[19], errors='coerce')

                    payment_map = {}
                    for _, row in payment_df.iterrows():
                        name = str(row[5]).strip().upper()
                        col_17 = row[17]
                        col_18 = row[18]
                        col_19 = row[19]
                        if pd.notnull(col_17) and col_17 > 0:
                            payment_map[name] = 'CASH'
                        elif pd.notnull(col_18) and col_18 > 0:
                            payment_map[name] = 'BANK'
                        elif pd.notnull(col_19) and col_19 > 0:
                            payment_map[name] = 'CARD'
                        else:
                            payment_map[name] = '-'

                    output_df['Payment'] = output_df['Name'].str.strip().str.upper().map(payment_map).fillna('CARD')

                    unmatched_names_df = output_df.copy()
                    unmatched_names_df['Payment'] = unmatched_names_df['Name'].str.strip().str.upper().map(payment_map)
                    
                    unmatched_names_df = unmatched_names_df[
                        ~((unmatched_names_df['Package'].str.contains('BODY COMPOSITION ANALYSIS', case=False, na=False)) & (unmatched_names_df['Price'] == 0))
                    ]
                    
                    unmatched_names = set(unmatched_names_df['Name'].str.strip().str.upper()) - set(payment_map.keys())
                    
                    if unmatched_names:
                        st.write("Unmatched Names (Filled as CARD):", list(unmatched_names))
                        st.write("Unmatched Names Count:", len(unmatched_names))

            except Exception as e:
                st.warning(f"⚠️ Error processing payment file: {e}. Proceeding without payment data.")

        # --- NEW AMENDMENT ---
        # Set 'Payment' to '-' for 'BODY COMPOSITION ANALYSIS' package
        output_df.loc[output_df['Package'].str.contains('BODY COMPOSITION ANALYSIS', case=False, na=False), 'Payment'] = '-'

        # Stage 2: Calculate hours
        hours_dict = {
            "UAE National Pre-employment": "72hrs", "Wellness Package - Premium": "96hrs",
            "Food Intolerance Test": "72hrs", "Respiratory Allergy Test": "48hrs",
            "Body Composition Analysis Test": "0", "ECG": "0","MOVEMENT ASSESSMENT": "0",
            "Wellness Package - Enhanced": "72hrs", "Wellness Package - Standard": "36hrs",
            "Lipid Profile Test": "24hrs", "Food Allergy Test": "48hrs",
            "Female Hormone Profile": "48hrs", "Gut Health": "6 Weeks",
            "Smart DNA - Nutrition Package": "6 Weeks",
            "SMART DNA - ACNE PACKAGE": "6 Weeks",
            "Smart DNA – Hair Loss Profile": "6 Weeks",
            "Smart DNA - Age Well Package": "6 Weeks",
            "ACTIVE PACKAGE": "96hrs",
            "ATHLETE PACKAGE": "96hrs",
            "MOVEMENT ASSESSMENT": "0",
            "BODY COMPOSITION ANALYSIS": "0",
            # FIX: Updated package name to remove apostrophe
            "PREMIUM PLUS MENS HEALTH SCREENING": "72hrs", 
            "GUT MICROBIOME PACKAGE": "6 Weeks",
            "PREMIUM HEALTH SCREENING": "72hrs",
            "STANDARD HEALTH SCREENING": "48hrs",
            # FIX: Updated package name to remove apostrophe
            "PREMIUM PLUS WOMENS HEALTH SCREENING": "72hrs", 
            "CLINICAL DIETITIAN PACKAGE - SINGLE": "0"
        }

        def get_hours(package_name):
            if not isinstance(package_name, str): return None
            if "SMART DNA" in package_name.upper(): return "6 Weeks"
            if "CONSULTATION" in package_name.upper(): return "0"
            if "MOVEMENT ASSESSMENT" in package_name.upper(): return "0"
            if "WOMENS COMPREHENSIVE HEALTH SCREENING" in package_name.upper(): return "96hrs"
            if "ACTIVE PACKAGE" in package_name.upper(): return "96hrs"
            if "ATHLETE PACKAGE" in package_name.upper(): return "96hrs"
            if "SEASONAL INFLUENZA" in package_name.upper(): return "0"
            if "BODY COMPOSITION ANALYSIS" in package_name.upper(): return "0"
            # FIX: Updated package name to remove apostrophe
            if "PREMIUM PLUS MENS HEALTH SCREENING" in package_name.upper(): return "72hrs"
            if "GUT MICROBIOME PACKAGE" in package_name.upper(): return "6 Weeks"
            if "PREMIUM HEALTH SCREENING" in package_name.upper(): return "72hrs"
            if "STANDARD HEALTH SCREENING" in package_name.upper(): return "48hrs"
            if "HEPATITIS B VACCINATION" in package_name.upper(): return "0"
            # FIX: Updated package name to remove apostrophe
            if "PREMIUM PLUS WOMENS HEALTH SCREENING" in package_name.upper(): return "72hrs"
            if "CLINICAL DIETITIAN PACKAGE - SINGLE" in package_name.upper(): return "0"
            if "CLINICAL DIETITIAN PACKAGE BUNDLE (X3)" in package_name.upper(): return "0"
            if "CLINICAL DIETITIAN PACKAGE BUNDLE (X5)" in package_name.upper(): return "0"
            if "FOOD ALLERGY & INTOLERANCE TEST BUNDLE" in package_name.upper(): return "96hrs"
            if "UAE NATIONAL PRE-EMPLOYMENT HEALTH SCREENING" in package_name.upper(): return "72hrs"
            if "FOOD INTOLERANCE TEST" in package_name.upper(): return "96hrs"
            if "SEASONAL FLU VACCINATION" in package_name.upper(): return "0"
            if "BCA - DISCOUNTED" in package_name.upper(): return "0"
            if "HEALTHY HEART PACKAGE" in package_name.upper(): return "72hrs"
            if "LIPID TEST" in package_name.upper(): return "48hrs"
            if "TRAVEL FIT ASSESSMENT" in package_name.upper(): return "0"
            for key in hours_dict:
                if key in package_name:
                    return hours_dict[key]
            return None

        output_df['Hrs'] = output_df['Package'].apply(get_hours)
        output_df = output_df.fillna('')

        # Stage 3: Pivot tables (DR 1 - QLAB)
        mapping_pivot = {
            "Wellness Package - Standard": "Standard Health Screening",
            "Wellness Package - Premium": "Premium Health Screening",
            "Respiratory Allergy Test (Add On)": "Respiratory Allergy Test",
            "Food Allergy Test (Add On)": "Food Allergy Test",
            "Food Intolerance Test (Add On)": "Food Intolerance Test",
            "Lipid Profile Test (Add On with Wellness)": "Lipid Test",
            "Smart DNA - Age Well Package": "Smart DNA - Age Well Package",
            "Smart DNA – Hair Loss Profile": "Smart DNA - Hair Loss Package",
            "Smart DNA – Acne Profile": "Smart DNA - Acne Package",
            "Womens Comprehensive Health Screening": "Women's Comprehensive Health Package",
            "Healthy Heart Package": "Healthy Heart Package",
            "UAE National Pre-employment": "UAE National Pre-Employment Health Screening"
        }
        packages_pivot = [
            'Standard Health Screening', 'Premium Health Screening', 'Premium PLUS Womens Health Screening',
            'Premium PLUS Mens Health Screening', 'Respiratory Allergy Test', 'Food Allergy Test',
            'Food Intolerance Test', 'Food Allergy & Intolerance Test Bundle', 'Coeliac Test',
            'Cortisol Test', 'Lipid Test', 'Smart DNA - Nutrition Package', 'Smart DNA - Age Well Package',
            'Smart DNA - Hair Loss Package', 'Smart DNA - Acne Package', 'Sports Health Screening Package',
            'Women\'s Comprehensive Health Package', 'Healthy Heart Package',
            'UAE National Pre-Employment Health Screening', 'Student Pre-Employment Health Screening'
        ]
        locations = ['CITY WALK', 'INDEX', 'DKP']

        output_df['Location'] = output_df['Location'].str.strip().str.upper()
        output_df['Package'] = output_df['Package'].str.strip().str.upper()

        pivot_df = pd.DataFrame(0, index=packages_pivot, columns=locations)

        for _, row in output_df.iterrows():
            package_value = str(row['Package']).upper()
            location_value = str(row['Location']).upper()
            mapped_package = next((mapping_pivot.get(k) for k in mapping_pivot.keys() if k.upper() in package_value), None)
            if mapped_package is None and any(pkg.upper() in package_value for pkg in packages_pivot):
                mapped_package = next(pkg for pkg in packages_pivot if pkg.upper() in package_value)
            if mapped_package in packages_pivot:
                if 'CITY WALK' in location_value:
                    pivot_df.at[mapped_package, 'CITY WALK'] += 1
                elif 'DUBAI KNOWLEDGE PARK' in location_value or 'DKP' in location_value:
                    pivot_df.at[mapped_package, 'DKP'] += 1
                elif 'INDEX TOWER' in location_value or 'INDEX' in location_value:
                    pivot_df.at[mapped_package, 'INDEX'] += 1

        # Add Total column
        pivot_df['Total'] = pivot_df.sum(axis=1)

        # Add Total row
        pivot_df.loc['Total'] = pivot_df.sum(axis=0)

        pivot_df = pivot_df.reindex(columns=['CITY WALK', 'INDEX', 'DKP', 'Total'])

        # Stage 4: Unique Patients (DR 1 - SS)
        # AMENDMENT START - New logic for handling BCA packages based on price
        
        # Define the packages list with the corrected names and the new row
        packages_unique = [
            'Standard Health Screening', 'Premium Health Screening', 'Premium PLUS Womens Health Screening',
            'Premium PLUS Mens Health Screening', 'Respiratory Allergy Test', 'Food Allergy Test',
            'Food Intolerance Test', 'Food Allergy & Intolerance Test Bundle', 'Coeliac Test',
            'Dietitian Panel WITH Coeliac - INTERNAL REFERRAL',
            'Cortisol Test', 'Lipid Test',
            'Gut Microbiome Package', 'Body Composition Analysis', 'BCA - Free', 'ECG & Consult',
            'Smart DNA - Nutrition Package', 'Smart DNA - Age Well Package', 'Smart DNA - Hair Loss Package',
            'Smart DNA - Acne Package', 'Doctor Consultation', 'Clinical Dietitian Package - Single',
            'Clinical Dietitian Package Bundle (x3)', 'Clinical Dietitian Package Bundle (x5)',
            'Sports Health Screening Package', 'Women\'s Comprehensive Health Package', 'Healthy Heart Package',
            'UAE National Pre-Employment Health Screening', 'Student Pre-Employment Health Screening',
            'Travel Fit Assessment', 'Seasonal Flu Vaccination', 'Hepatitis B Vaccination'
        ]

        temp_df = output_df.copy()

        # FIX: New logic for BCA counting
        def get_standardized_package(row):
            pkg_name = str(row['Package']).strip().upper()
            price = row['Price']
            
            # Special case for BCA packages
            if 'BODY COMPOSITION ANALYSIS' in pkg_name or 'BCA - DISCOUNTED' in pkg_name:
                # If price is 0, it's a free BCA
                if price == 0:
                    return 'BCA - Free'
                # If price is > 0, it's a regular BCA
                else:
                    return 'Body Composition Analysis'

            # General mappings
            mapping_unique = {
                "WELLNESS PACKAGE - STANDARD": "Standard Health Screening",
                "WELLNESS PACKAGE - PREMIUM": "Premium Health Screening",
                "RESPIRATORY ALLERGY TEST (STAND ALONE)": "Respiratory Allergy Test",
                "FOOD ALLERGY TEST (STAND ALONE)": "Food Allergy Test",
                "FOOD INTOLERANCE TEST (STAND ALONE)": "Food Intolerance Test",
                "COELIAC PROFILE TEST (STAND ALONE)": "Coeliac Test",
                "LIPID PROFILE TEST (STAND ALONE)": "Lipid Test",
                "GUT HEALTH (STAND ALONE)": "Gut Microbiome Package",
                "ECG AND DOCTOR CONSULT (STAND ALONE)": "ECG & Consult",
                "SMART DNA - NUTRITION PACKAGE": "Smart DNA - Nutrition Package",
                "SMART DNA - AGE WELL PACKAGE": "Smart DNA - Age Well Package",
                "SMART DNA – HAIR LOSS PROFILE": "Smart DNA - Hair Loss Package",
                "SMART DNA – ACNE PACKAGE": "Smart DNA - Acne Package",
                "OUTPATIENT CONSULTATION - 30 MINS": "Doctor Consultation",
                "CLINICAL DIETICIAN PROFILE TEST": "Clinical Dietitian Package - Single",
                "WOMENS COMPREHENSIVE HEALTH SCREENING": "Women's Comprehensive Health Package",
                "HEALTHY HEART PACKAGE": "Healthy Heart Package",
                "UAE NATIONAL PRE-EMPLOYMENT": "UAE National Pre-Employment Health Screening",
                "TRAVEL FIT ASSESSMENT": "Travel Fit Assessment",
                "SEASONAL INFLUENZA (STAND ALONE)": "Seasonal Flu Vaccination",
                "PREMIUM PLUS MENS HEALTH SCREENING": "Premium PLUS Mens Health Screening",
                "PREMIUM PLUS WOMENS HEALTH SCREENING": "Premium PLUS Womens Health Screening",
                "GUT MICROBIOME PACKAGE": "Gut Microbiome Package",
                "DIETITIAN PANEL WITH COELIAC - INTERNAL REFERRAL": "Dietitian Panel WITH Coeliac - INTERNAL REFERRAL",
            }

            # Check if the package name is in the mapping dictionary
            if pkg_name in mapping_unique:
                return mapping_unique[pkg_name]
            
            # Check for substring matches in the packages_unique list
            for pkg in packages_unique:
                if pkg.upper() in pkg_name:
                    return pkg
            
            return None

        # Apply the new mapping logic
        temp_df['Standardized Package'] = temp_df.apply(get_standardized_package, axis=1)

        # Filter out rows where package could not be matched
        temp_df = temp_df[temp_df['Standardized Package'].notna()]
        
        # Create an empty DataFrame to store the unique patient counts
        unique_patients_pivot = pd.DataFrame(0, index=packages_unique, columns=locations)

        # Group and count unique patients for all packages
        grouped_data = temp_df.groupby(['Standardized Package', 'Location'])['Name'].nunique()

        # Populate the unique_patients_pivot DataFrame from the grouped data
        for (package, location), count in grouped_data.items():
            if package in unique_patients_pivot.index:
                if 'CITY WALK' in location.upper():
                    unique_patients_pivot.at[package, 'CITY WALK'] += count
                elif 'DUBAI KNOWLEDGE PARK' in location.upper() or 'DKP' in location.upper():
                    unique_patients_pivot.at[package, 'DKP'] += count
                elif 'INDEX TOWER' in location.upper() or 'INDEX' in location.upper():
                    unique_patients_pivot.at[package, 'INDEX'] += count
        
        # Add Total column
        unique_patients_pivot['Total'] = unique_patients_pivot.sum(axis=1)

        # Add Total row
        unique_patients_pivot.loc['Total'] = unique_patients_pivot.sum(axis=0)

        # Add a row for total unique patients per location
        unique_patients_total_row = pd.Series(index=locations + ['Total'], dtype=int)
        for location in locations:
            if location == 'CITY WALK':
                mask = output_df['Location'].str.contains('CITY WALK', case=False, na=False)
            elif location == 'DKP':
                mask = output_df['Location'].str.contains('DUBAI KNOWLEDGE PARK|DKP', case=False, na=False)
            elif location == 'INDEX':
                mask = output_df['Location'].str.contains('INDEX TOWER|INDEX', case=False, na=False)
            else:
                continue
            
            unique_count = output_df[mask]['Name'].nunique()
            unique_patients_total_row[location] = unique_count
        
        unique_patients_total_row['Total'] = unique_patients_total_row[locations].sum()
        unique_patients_pivot.loc['Unique Patients'] = unique_patients_total_row
        
        unique_patients_pivot = unique_patients_pivot.reindex(columns=['CITY WALK', 'INDEX', 'DKP', 'Total'])
        # AMENDMENT END

        # Stage 5: Filtered table for QLAB
        stage4_df = output_df.copy()
        # FIX: Added filter for 'BCA - DISCOUNTED'
        stage4_df = stage4_df[~stage4_df['Package'].str.contains('GUT HEALTH|CONSULTATION|GUT MICROBIOME PACKAGE|BCA - DISCOUNTED', case=False, na=False)]
        stage4_df = stage4_df[stage4_df['Hrs'] != '0']
        stage4_df['No'] = range(1, len(stage4_df) + 1)
        stage4_df = stage4_df.fillna('')

        # Store DataFrames in session state
        st.session_state.output_df = output_df
        st.session_state.pivot_df = pivot_df
        st.session_state.unique_patients_pivot = unique_patients_pivot

        # Write to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pivot_df.to_excel(writer, sheet_name='DR 1 - QLAB', index=True)
            unique_patients_pivot.to_excel(writer, sheet_name='DR 1 - SS', index=True)
            output_df.to_excel(writer, sheet_name='DR 2', index=False)
            stage4_df.to_excel(writer, sheet_name='DR QLAB', index=False)

            workbook = writer.book
            header_format = workbook.add_format({
                'bg_color': '#00899D',
                'font_color': 'white',
                'bold': True,
                'font_size': 14,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            second_row_format = workbook.add_format({
                'bg_color': '#00899D',
                'font_color': 'white',
                'bold': True,
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            record_format = workbook.add_format({
                'bg_color': '#EEECE1',
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            total_format = workbook.add_format({
                'bg_color': '#00899D',
                'font_color': 'white',
                'bold': True,
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })

            def auto_adjust_column_width(worksheet, df, start_col=0, index=False):
                """Adjust column widths based on content length."""
                for col_num, column in enumerate(df.columns, start_col):
                    max_length = max(
                        len(str(column)) + 2,
                        max((len(str(val)) for val in df[column]), default=0) + 2
                    )
                    if index and col_num == start_col:
                        max_length = max(
                            len(str(df.index.name or '')) + 2,
                            max((len(str(idx)) for idx in df.index), default=0) + 2
                        )
                    worksheet.set_column(col_num, col_num, max_length)

            # Formatting for DR 1 - QLAB
            worksheet_qlab_pivot = writer.sheets['DR 1 - QLAB']
            worksheet_qlab_pivot.set_column('A:A', 40.44)
            worksheet_qlab_pivot.set_column('B:E', 10)
            worksheet_qlab_pivot.set_row(0, 32.4)
            worksheet_qlab_pivot.merge_range('A1:E1', header_date, header_format)
            worksheet_qlab_pivot.write(1, 0, '', second_row_format)
            for col_num, value in enumerate(pivot_df.columns.values):
                worksheet_qlab_pivot.write(1, col_num + 1, value, second_row_format)
            for row_num in range(len(pivot_df)):
                for col_num, value in enumerate(pivot_df.iloc[row_num]):
                    format_to_use = total_format if (row_num == len(pivot_df) - 1 or col_num == len(pivot_df.columns) - 1) else record_format
                    worksheet_qlab_pivot.write(row_num + 2, col_num + 1, value, format_to_use)
                format_to_use = total_format if row_num == len(pivot_df) - 1 else second_row_format
                worksheet_qlab_pivot.write(row_num + 2, 0, pivot_df.index[row_num], format_to_use)

            # Formatting for DR 1 - SS
            worksheet_ss_pivot = writer.sheets['DR 1 - SS']
            worksheet_ss_pivot.set_column('A:A', 40.44)
            worksheet_ss_pivot.set_column('B:E', 10)
            worksheet_ss_pivot.set_row(0, 32.4)
            worksheet_ss_pivot.merge_range('A1:E1', header_date, header_format)
            worksheet_ss_pivot.write(1, 0, '', second_row_format)
            for col_num, value in enumerate(unique_patients_pivot.columns.values):
                worksheet_ss_pivot.write(1, col_num + 1, value, second_row_format)
            for row_num in range(len(unique_patients_pivot)):
                for col_num, value in enumerate(unique_patients_pivot.iloc[row_num]):
                    # Check if the value is an Excel formula string
                    if isinstance(value, str) and value.startswith('='):
                        format_to_use = total_format
                    else:
                        format_to_use = total_format if (row_num >= len(unique_patients_pivot) - 2 or col_num == len(unique_patients_pivot.columns) - 1) else record_format
                    worksheet_ss_pivot.write(row_num + 2, col_num + 1, value, format_to_use)
                
                format_to_use = total_format if row_num >= len(unique_patients_pivot) - 2 else second_row_format
                worksheet_ss_pivot.write(row_num + 2, 0, unique_patients_pivot.index[row_num], format_to_use)


            # Formatting for DR 2
            worksheet = writer.sheets['DR 2']
            auto_adjust_column_width(worksheet, output_df)
            worksheet.set_column('D:D', 32.35)
            worksheet.merge_range('A1:L1', header_date, header_format)
            for col_num, value in enumerate(output_df.columns.values):
                worksheet.write(1, col_num, value, second_row_format)
            for row_num in range(len(output_df)):
                for col_num, value in enumerate(output_df.iloc[row_num]):
                    worksheet.write(row_num + 2, col_num, value, record_format)

            # Formatting for DR QLAB
            worksheet_qlab = writer.sheets['DR QLAB']
            auto_adjust_column_width(worksheet_qlab, stage4_df)
            worksheet_qlab.set_column('D:D', 32.35)
            worksheet_qlab.merge_range('A1:L1', header_date, header_format)
            for col_num, value in enumerate(stage4_df.columns.values):
                worksheet_qlab.write(1, col_num, value, second_row_format)
            for row_num in range(len(stage4_df)):
                for col_num, value in enumerate(stage4_df.iloc[row_num]):
                    worksheet_qlab.write(row_num + 2, col_num, value, record_format)

        st.success("✅ File processed successfully!")
        
        # Create two columns for buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                label="📥 Download Final Output",
                data=output.getvalue(),
                file_name="final_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_button",
                help="Download the processed Excel file",
                use_container_width=True
            )
        with col2:
            if st.button(
                label="📊 Analytics",
                key="analytics_button",
                help="View analytics and charts",
                use_container_width=True
            ):
                st.switch_page("pages/analyticss.py")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")

# ---------- Footer ----------
load_footer()
