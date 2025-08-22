import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from style_utils import load_css, load_footer # Import the shared functions
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
                if payment_df.shape[1] < 19:
                    raise ValueError("Payment file must have at least 19 columns (0-18).")
                payment_df = payment_df.iloc[:, [5, 17, 18]]
                payment_df = payment_df.dropna(subset=[5]).reset_index(drop=True)
                payment_df[17] = pd.to_numeric(payment_df[17], errors='coerce')
                payment_df[18] = pd.to_numeric(payment_df[18], errors='coerce')

                payment_map = {}
                for _, row in payment_df.iterrows():
                    name = str(row[5]).strip().upper()
                    col_17 = row[17]
                    col_18 = row[18]
                    if pd.notnull(col_17) and col_17 > 0:
                        payment_map[name] = 'CASH'
                    elif pd.notnull(col_18) and col_18 > 0:
                        payment_map[name] = 'BANK'
                    else:
                        payment_map[name] = 'CARD'

                output_df['Payment'] = output_df['Name'].str.strip().str.upper().map(payment_map).fillna('CARD')

                # Show unmatched names if they exist
                unmatched_names = set(output_df['Name'].str.strip().str.upper()) - set(payment_map.keys())
                if unmatched_names:
                    st.write("Unmatched Names (Filled as CARD):", list(unmatched_names))
                    st.write("Unmatched Names Count:", len(unmatched_names))

            except Exception as e:
                st.warning(f"⚠️ Error processing payment file: {e}. Proceeding without payment data.")

        # Stage 2: Calculate hours
        hours_dict = {
            "UAE National Pre-employment": "72hrs", "Wellness Package - Premium": "96hrs",
            "Food Intolerance Test": "96hrs", "Respiratory Allergy Test": "48hrs",
            "Body Composition Analysis Test": "0", "ECG": "0","MOVEMENT ASSESSMENT": "0",
            "Wellness Package - Enhanced": "72hrs", "Wellness Package - Standard": "36hrs",
            "Lipid Profile Test": "24hrs", "Food Allergy Test": "48hrs",
            "Female Hormone Profile": "48hrs", "Gut Health": "6 Weeks",
            "Smart DNA - Nutrition Package": "6 Weeks",
            "Smart DNA – Acne Profile": "6 Weeks",
            "Smart DNA – Hair Loss Profile": "6 Weeks",
            "Smart DNA - Age Well Package": "6 Weeks",
            "ACTIVE PACKAGE": "96hrs",
            "ATHLETE PACKAGE": "96hrs",
            "MOVEMENT ASSESSMENT": "0"
        }

        def get_hours(package_name):
            if not isinstance(package_name, str): return None
            if "SMART DNA - " in package_name: return "6 Weeks"
            if "CONSULTATION" in package_name.upper(): return "0"
            if "MOVEMENT ASSESSMENT" in package_name.upper(): return "0"
            if "WOMENS COMPREHENSIVE HEALTH SCREENING" in package_name.upper(): return "96hrs"
            if "ACTIVE PACKAGE" in package_name.upper(): return "96hrs"
            if "ATHLETE PACKAGE" in package_name.upper(): return "96hrs"
            if "SEASONAL INFLUENZA" in package_name.upper(): return "0" 
            for key in hours_dict:
                if key in package_name:
                    return hours_dict[key]
            return None

        output_df['Hrs'] = output_df['Package'].apply(get_hours)
        output_df = output_df.fillna('')

        # Stage 3: Pivot tables (DR 1 - QLAB)
        mapping_pivot = {
            "UAE National Pre-employment": "UAE-National Pre-Employment Test",
            "Wellness Package - Premium": "Premium Package",
            "Food Intolerance Test (Stand Alone)": "Food Intolerance",
            "Respiratory Allergy Test (Add On)": "Respiratory Allergy",
            "Body Composition Analysis Test (Add On)": "Body Composition Analysis Test (Add On)",
            "ECG and Doctor Consult (Stand Alone)": "ECG and Doctor Consult (Stand Alone)",
            "Wellness Package - Enhanced": "Enhanced Package",
            "Wellness Package - Standard": "Standard Package",
            "Lipid Profile Test (Add On with Wellness)": "Lipid Profile",
            "Food Allergy Test (Add On)": "Food Allergy",
            "Female Hormone Profile (Add On with Wellness)": "Female Hormone Profile",
            "Food Intolerance Test (Add On)": "Food Intolerance",
            "Smart DNA - Age Well Package": "Age-Well"
        }
        packages_pivot = [
            'Standard Package', 'Enhanced Package', 'Premium Package', 'Lipid Profile',
            'Food Allergy', 'Food Intolerance', 'Respiratory Allergy', 'Female Hormone Profile',
            'Mag & Zinc', 'Coeliac Profile Test', 'Active Package',
            'Womens Comprehensive Health Screening', 'Healthy Heart Package', 'Right Fit',
            'Athlete Package', 'Nutrition', 'UAE-National Pre-Employment Test', 'Age-Well',
            'Acne Profile', 'Hair Loss'
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
        packages_unique = [
            'Standard Package', 'Enhanced Package', 'Premium Package', 'Lipid Profile',
            'Food Allergy', 'Food Intolerance', 'Respiratory Allergy', 'Female Hormone Profile',
            'Mag & Zinc', 'Coeliac Profile Test', 'Active Package', 'Athlete Package',
            'BCA', 'Right Fit', 'ECG', 'Pulmonary Function Test',
            'UAE-National Pre-Employment Test', 'Travel Fit Assessment', 'Movement Assessment',
            'H&U Vaccination', 'Influenza Vaccination', 'Healthy Heart',
            'Womens Comprehensive Health Screening', 'Gym Partnership Package - Athlete Plus',
            'Nutrition', 'Age-Well', 'Acne Profile', 'Hair Loss', 'GUT Health', 'OPC'
        ]

        mapping_unique = {
            "UAE National Pre-employment": "UAE-National Pre-Employment Test",
            "Wellness Package - Premium": "Premium Package",
            "Food Intolerance Test (Stand Alone)": "Food Intolerance",
            "Food Intolerance Test (Add On)": "Food Intolerance",
            "Respiratory Allergy Test (Add On)": "Respiratory Allergy",
            "Body Composition Analysis Test": "BCA",
            "Body Composition Analysis Test (Add On)": "BCA",
            "ECG and Doctor Consult (Stand Alone)": "ECG",
            "Wellness Package - Enhanced": "Enhanced Package",
            "Wellness Package - Standard": "Standard Package",
            "Lipid Profile Test (Add On with Wellness)": "Lipid Profile",
            "Food Allergy Test (Add On)": "Food Allergy",
            "Female Hormone Profile (Add On with Wellness)": "Female Hormone Profile",
            "Smart DNA - Age Well Package": "Age-Well",
            "Gut Health": "GUT Health",
            "Healthy Heart Package": "Healthy Heart",
            "Womens Comprehensive Health Screening": "Womens Comprehensive Health Screening",
            "Athlete Package": "Athlete Package",
            "Right Fit": "Right Fit",
            "Smart DNA - Nutrition Package": "Nutrition",
            "Acne Profile": "Acne Profile",
            "Hair Loss": "Hair Loss",
            "Coeliac Profile Test": "Coeliac Profile Test",
            "Active Package": "Active Package",
            "Mag & Zinc": "Mag & Zinc",
            "Pulmonary Function Test": "Pulmonary Function Test",
            "Travel Fit Assessment": "Travel Fit Assessment",
            "Movement Assessment": "Movement Assessment",
            "H&U Vaccination": "H&U Vaccination",
            "SEASONAL INFLUENZA": "Influenza Vaccination",
            "Gym Partnership Package - Athlete Plus": "Gym Partnership Package - Athlete Plus",
            "CLINICAL DIETITIAN CONSULTATION - 30 MINS": "OPC",
            "Outpatient Consultation - 30 Mins": "OPC"
        }

        unique_patients_pivot = pd.DataFrame(0, index=packages_unique, columns=locations)

        def match_package(service_name):
            if not isinstance(service_name, str):
                return None
            service_name = service_name.strip().upper()
            for key, value in mapping_unique.items():
                if key.upper() == service_name:
                    return value
            for key, value in mapping_unique.items():
                if key.upper() in service_name:
                    return value
            for pkg in packages_unique:
                if pkg.upper() in service_name:
                    return pkg
            return None

        for _, row in output_df.iterrows():
            package_value = match_package(row['Package'])
            location_value = str(row['Location']).strip().upper()
            if package_value in packages_unique:
                if 'CITY WALK' in location_value:
                    unique_patients_pivot.at[package_value, 'CITY WALK'] += 1
                elif 'DUBAI KNOWLEDGE PARK' in location_value or 'DKP' in location_value:
                    unique_patients_pivot.at[package_value, 'DKP'] += 1
                elif 'INDEX TOWER' in location_value or 'INDEX' in location_value:
                    unique_patients_pivot.at[package_value, 'INDEX'] += 1

        # Add Total column for packages
        unique_patients_pivot['Total'] = unique_patients_pivot.sum(axis=1)

        # Add Total row for packages
        unique_patients_pivot.loc['Total'] = unique_patients_pivot.sum(axis=0)

        

        # Add Unique Patients row after Total (not included in Total calculation)
        unique_patients_row = pd.Series(index=locations + ['Total'], dtype=int)
        for location in locations:
            if location == 'CITY WALK':
                mask = output_df['Location'].str.contains('CITY WALK', case=False, na=False)
            elif location == 'DKP':
                mask = output_df['Location'].str.contains('DUBAI KNOWLEDGE PARK|DKP', case=False, na=False)
            elif location == 'INDEX':
                mask = output_df['Location'].str.contains('INDEX TOWER|INDEX', case=False, na=False)
            unique_count = output_df[mask]['Name'].nunique()
            unique_patients_row[location] = unique_count
        unique_patients_row['Total'] = unique_patients_row[locations].sum()

        # Append Unique Patients row
        unique_patients_pivot.loc['Unique Patients'] = unique_patients_row

        unique_patients_pivot = unique_patients_pivot.reindex(columns=['CITY WALK', 'INDEX', 'DKP', 'Total'])
        # add beautician 

         # Prepare Excel formula for 4th column (index 3)
        beautician_excel_row = unique_patients_pivot.shape[0] + 3  # 1-based + header + column names
        
        beautician_row = pd.DataFrame(
            [['N/A', 'N/A', 0, f'=D{beautician_excel_row}']],
            columns=unique_patients_pivot.columns,
            index=['Beautician']
        )
        
        # Add Beautician at the end
        unique_patients_pivot = pd.concat([unique_patients_pivot, beautician_row])

        # Stage 5: Filtered table for QLAB
        stage4_df = output_df.copy()
        stage4_df = stage4_df[~stage4_df['Package'].str.contains('GUT HEALTH|CONSULTATION', case=False, na=False)]
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
                        len(str(column)) + 2,  # Column header length
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
                    format_to_use = total_format if (row_num >= len(unique_patients_pivot) - 3 or col_num == len(unique_patients_pivot.columns) - 1) else record_format
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
