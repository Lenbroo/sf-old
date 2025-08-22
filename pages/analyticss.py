import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import re
from pyvis.network import Network
import streamlit.components.v1 as components

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Smart Flow Analytics", layout="wide")

# This CSS combines the original needs with the new, enhanced styling.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    .stApp {
        background-image: url("https://analytics.smartsalem.tech/smartsuite/bg-login.688761c8fcdbbeb1.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        min-height: 100vh;
    }
            /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Expand main content */
        [data-testid="stAppViewContainer"] {
            margin-left: 0;
        }
        /* Hide top header */
        header[data-testid="stHeader"] {
            display: none;
        }
        /* Hide profile menu and other floating buttons bottom-right */
        .css-1v3fvcr.e1fqkh3o3,  /* Common profile button container class, may vary */
        [data-testid="stToolbar"],
        button[aria-label="User menu"],
        div[role="complementary"] {
            display: none !important;
        }
    h1, h2, h3, h4, h5, .stMarkdown, .stMarkdown div, .stMarkdown p, .stMetric label, .stMetric div {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
        font-family: 'Roboto', sans-serif !important;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        text-align: center !important;
        min-height: 120px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: 10px !important;
    }
    
    .stDataFrame, .stTable {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-divider {
        border-top: 2px solid #00899D;
        margin: 40px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def style_chart(fig, title):
    """Applies consistent styling to a Plotly figure."""
    fig.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0.2)',
        font_color='white',
        font_family='Roboto',
        title_font_size=20,
        legend_title_font_color='white',
        xaxis=dict(title_font_color='white', tickfont_color='white', gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title_font_color='white', tickfont_color='white', gridcolor='rgba(255, 255, 255, 0.1)'),
        margin=dict(t=60, b=50, l=50, r=50),
        showlegend=True
    )
    return fig

def format_number(x):
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x)
    return x

def load_footer():
    """Loads the shared footer for the application."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://i.ibb.co/GQkBhDWj/Image.png" width="250"/>
            <div style="margin-top: 10px; color: white; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);">
                <sub>Made with ❤️ by Khaled Abdelhamid</sub><br>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("📊 Smart Flow Analytics Dashboard")
st.markdown("Explore comprehensive and deep insights from our wellness sales report.", unsafe_allow_html=True)

# Check if DataFrames exist in session state
if ('output_df' not in st.session_state or st.session_state.output_df is None or
    'pivot_df' not in st.session_state or st.session_state.pivot_df is None or
    'unique_patients_pivot' not in st.session_state or st.session_state.unique_patients_pivot is None):
    st.error("⚠️ No data available. Please upload a file on the main page first.")
    st.stop()

# --- Data Retrieval and Preparation ---
output_df = st.session_state.output_df.copy()
pivot_df = st.session_state.pivot_df.copy()
unique_patients_pivot = st.session_state.unique_patients_pivot.copy()

# NEW LINE: Exclude 'UAE NATIONAL PRE-EMPLOYMENT' packages from all analysis
# هذا السطر يستثني باقات "UAE NATIONAL PRE-EMPLOYMENT" من جميع التحليلات
output_df = output_df[output_df['Package'] != 'UAE NATIONAL PRE-EMPLOYMENT']

output_df['Package'] = output_df['Package'].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
output_df['Date'] = pd.to_datetime(output_df['Date'], format='%d/%m/%Y', errors='coerce')


# --- Analytics Sections ---

# ---------- Overview Section ----------
st.markdown("<h3>Overview</h3>", unsafe_allow_html=True)
total_records = len(output_df)
total_unique_patients = output_df['Name'].nunique()
total_revenue = output_df['Price'].sum()
avg_revenue_per_patient = total_revenue / total_unique_patients if total_unique_patients > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric("Total sold packages", f"{total_records:,}")
with col2:
    st.metric("Unique Customers", f"{total_unique_patients:,}")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Revenue (AED)", f"{total_revenue:,.2f}")
with col2:
    st.metric("Avg Revenue per Customer (AED)", f"{avg_revenue_per_patient:,.2f}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Customer Behavior Analysis (Moved to a rational place) ----------
st.markdown("<h3>Customer Behavior Analysis (Visit on Different Day)</h3>", unsafe_allow_html=True)
if not output_df.empty:
    # A returning customer is one with visits on more than one unique day
    # Note: The main filtering line above already excludes the unwanted package.
    customer_visits = output_df.groupby('Name')['Date'].nunique()
    returning_customers_names = customer_visits[(customer_visits > 1)].index
    
    returning_customers_count = len(returning_customers_names)
    new_customers_count = total_unique_patients - returning_customers_count

    returning_df = output_df[output_df['Name'].isin(returning_customers_names)]
    new_df = output_df[~output_df['Name'].isin(returning_customers_names)]

    returning_revenue = returning_df['Price'].sum()
    new_revenue = new_df['Price'].sum()
    avg_returning_revenue = returning_revenue / returning_customers_count if returning_customers_count > 0 else 0
    avg_new_revenue = new_revenue / new_customers_count if new_customers_count > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Returning Customers (Different Days)", f"{returning_customers_count:,}", f"Total Revenue: {returning_revenue:,.0f} AED")
        st.metric("Avg. Revenue / Returning Customer", f"{avg_returning_revenue:,.0f} AED")
    with col2:
        st.metric("New Customers (Single Visit)", f"{new_customers_count:,}", f"Total Revenue: {new_revenue:,.0f} AED")
        st.metric("Avg. Revenue / New Customer", f"{avg_new_revenue:,.0f} AED")


    # Scrollable table for returning clients
    st.markdown("<h5>Returning Customer Details</h5>", unsafe_allow_html=True)
    returning_details = output_df[
    (output_df['Name'].isin(returning_customers_names))][['Name', 'Date', 'Package', 'Price']].sort_values(by=['Name', 'Date'])
    st.dataframe(returning_details, height=300, use_container_width=True) # Makes the dataframe scrollable
else:
    st.info("Not enough data for customer behavior analysis.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ---------- Package Insights Section ----------
st.markdown("<h3>Package Insights</h3>", unsafe_allow_html=True)
# Ensure pivot_df and unique_patients_pivot are also re-filtered to be consistent.
# The original script does this implicitly by re-generating the pivot tables in app.py
# but for safety and clarity in this file, we'll re-filter if needed.
# Since app.py handles pivot tables, we will assume they are already filtered correctly,
# but we will make sure the other analyses are also filtered.

# Update pivot table to exclude the package from analysis
pivot_df_filtered = st.session_state.pivot_df.copy()
if 'UAE-National Pre-Employment Test' in pivot_df_filtered.index:
    pivot_df_filtered = pivot_df_filtered.drop('UAE-National Pre-Employment Test', errors='ignore')
if 'UAE NATIONAL PRE-EMPLOYMENT' in pivot_df_filtered.index:
    pivot_df_filtered = pivot_df_filtered.drop('UAE NATIONAL PRE-EMPLOYMENT', errors='ignore')

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h5>Top 5 Packages by Count</h5>", unsafe_allow_html=True)
    # FIX: Added a check for 'Total' column to prevent KeyError
    if 'Total' in pivot_df_filtered.columns:
        top_packages = pivot_df_filtered.drop('Total', axis=0, errors='ignore')['Total'].sort_values(ascending=False).head(5)
        st.dataframe(top_packages[top_packages > 0].map(format_number), use_container_width=True)
    else:
        st.info("The 'Total' column is missing from the pivot table data.")
with col2:
    st.markdown("<h5>Top 5 Packages by Revenue</h5>", unsafe_allow_html=True)
    package_revenue = output_df.groupby('Package')['Price'].sum().sort_values(ascending=False).head(5)
    st.dataframe(package_revenue[package_revenue > 0].map(format_number), use_container_width=True)

# Packages with Zero Sales
st.markdown("<h5>Packages with Zero Sales</h5>", unsafe_allow_html=True)
zero_sales = unique_patients_pivot[unique_patients_pivot['Total'] == 0].index.tolist()
# Filter out the excluded package from the zero sales list as well
zero_sales = [pkg for pkg in zero_sales if pkg != 'Total' and pkg != 'Unique Patients' and pkg != 'UAE-National Pre-Employment Test' and pkg != 'UAE NATIONAL PRE-EMPLOYMENT']
if zero_sales:
    zero_sales_html = "<ul style='list-style-type: disc; padding-left: 20px;'>" + "".join(f"<li>{pkg}</li>" for pkg in zero_sales) + "</ul>"
    st.markdown(
        f"""
        <div style='background-color: rgba(0, 0, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 4px solid #00899D;'>
            {zero_sales_html}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("All packages had sales during this period.")# Bar Chart: Package Distribution by Location
st.markdown("<h5>Package Distribution by Location</h5>", unsafe_allow_html=True)
plot_data = pivot_df_filtered.drop('Total', axis=0, errors='ignore').reset_index().melt(
    id_vars='index', value_vars=['CITY WALK', 'INDEX', 'DKP'], var_name='Location', value_name='Count'
)
plot_data = plot_data[plot_data['Count'] > 0]
fig1 = px.bar(
    plot_data, x='index', y='Count', color='Location', barmode='group',
    color_discrete_map={'CITY WALK': '#00899D', 'INDEX': '#FF6F61', 'DKP': '#6B7280'}
)
fig1 = style_chart(fig1, "Package Distribution by Location")
fig1.update_layout(xaxis_title="Package", yaxis_title="Number of sold packages", xaxis_tickangle=45)
st.plotly_chart(fig1, use_container_width=True)

# Bar Chart: Top 5 Packages by Revenue
st.markdown("<h5>Top 5 Packages by Revenue</h5>", unsafe_allow_html=True)
fig2 = px.bar(
    package_revenue.reset_index(), x='Package', y='Price',
    labels={'Price': 'Revenue (AED)'}, color_discrete_sequence=['#00899D']
)
fig2 = style_chart(fig2, "Top 5 Packages by Revenue (AED)")
fig2.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Location Insights Section ----------
st.markdown("<h3>Location Insights</h3>", unsafe_allow_html=True)
# All calculations below will now use the filtered `output_df`
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h5>sold packages by Location</h5>", unsafe_allow_html=True)
    location_counts = output_df['Location'].value_counts().sort_values(ascending=False)
    st.dataframe(location_counts.map(format_number), use_container_width=True)
with col2:
    st.markdown("<h5>Revenue by Location</h5>", unsafe_allow_html=True)
    location_revenue = output_df.groupby('Location')['Price'].sum().sort_values(ascending=False)
    st.dataframe(location_revenue.map(format_number), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h5>Revenue Distribution by Location</h5>", unsafe_allow_html=True)
    fig3 = px.pie(
        location_revenue.reset_index(), names='Location', values='Price',
        color_discrete_sequence=['#00899D', '#FF6F61', '#6B7280']
    )
    fig3 = style_chart(fig3, "Revenue Distribution by Location")
    st.plotly_chart(fig3, use_container_width=True)
with col2:
    st.markdown("<h5>Unique Customers by Location</h5>", unsafe_allow_html=True)
    # Re-calculate unique patients based on the filtered output_df
    locations = ['CITY WALK', 'INDEX', 'DKP']
    unique_patients_row = pd.Series(index=locations, dtype=int)
    for location in locations:
        if location == 'CITY WALK':
            mask = output_df['Location'].str.contains('CITY WALK', case=False, na=False)
        elif location == 'DKP':
            mask = output_df['Location'].str.contains('DUBAI KNOWLEDGE PARK|DKP', case=False, na=False)
        elif location == 'INDEX':
            mask = output_df['Location'].str.contains('INDEX TOWER|INDEX', case=False, na=False)
        unique_count = output_df[mask]['Name'].nunique()
        unique_patients_row[location] = unique_count

    fig4 = go.Figure(data=[go.Bar(x=unique_patients_row.index, y=unique_patients_row.values, marker_color='#00899D')])
    fig4 = style_chart(fig4, "Unique Customers by Location")
    fig4.update_layout(xaxis_title="Location", yaxis_title="Number of Unique Customers")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Time Trends Section ----------
st.markdown("<h3>Time Trends Analysis</h3>", unsafe_allow_html=True)
valid_dates_df = output_df[output_df['Date'].notna()]
time_data = pd.DataFrame() # Initialize empty dataframe
if not valid_dates_df.empty:
    time_data = valid_dates_df.groupby(valid_dates_df['Date'].dt.date).agg(Records=('No', 'count'), Revenue=('Price', 'sum')).reset_index()
    time_data.columns = ['Date', 'Records', 'Revenue']
    time_data['Date'] = pd.to_datetime(time_data['Date'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h5>Peak Days (by sold packages)</h5>", unsafe_allow_html=True)
        peak_records = time_data.nlargest(3, 'Records')[['Date', 'Records']].reset_index(drop=True)
        st.dataframe(peak_records, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("<h5>Peak Days (by Revenue)</h5>", unsafe_allow_html=True)
        peak_revenue = time_data.nlargest(3, 'Revenue')[['Date', 'Revenue']].reset_index(drop=True)
        st.dataframe(peak_revenue, use_container_width=True, hide_index=True)

    st.markdown("<h5>sold packages Over Time</h5>", unsafe_allow_html=True)
    fig5 = px.line(time_data, x='Date', y='Records', labels={'Records': 'Number of sold packages'}, color_discrete_sequence=['#00899D'])
    fig5 = style_chart(fig5, "sold packages Over Time")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<h5>Revenue Over Time</h5>", unsafe_allow_html=True)
    fig6 = px.line(time_data, x='Date', y='Revenue', labels={'Revenue': 'Revenue (AED)'}, color_discrete_sequence=['#FF6F61'])
    fig6 = style_chart(fig6, "Revenue Over Time")
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("No valid date data to display time trends.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Day and Time Trends Section ----------
st.markdown("<h3>Day and Hour Analysis</h3>", unsafe_allow_html=True)
if not valid_dates_df.empty:
    valid_dates_df['DayOfWeek'] = valid_dates_df['Date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = valid_dates_df.groupby('DayOfWeek').agg({'No': 'count', 'Price': 'sum'}).reindex(day_order).reset_index()
    day_data.columns = ['Day', 'Records', 'Revenue']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h5>Days by Number of sold packages</h5>", unsafe_allow_html=True)
        st.dataframe(day_data[['Day', 'Records']].sort_values(by='Records', ascending=False), use_container_width=True)
    with col2:
        st.markdown("<h5>Days by Revenue</h5>", unsafe_allow_html=True)
        st.dataframe(day_data[['Day', 'Revenue']].sort_values(by='Revenue', ascending=False), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h5>sold packages by Day of Week</h5>", unsafe_allow_html=True)
        fig_day_records = px.bar(day_data, x='Day', y='Records', labels={'Records': 'Number of sold packages'}, color_discrete_sequence=['#00899D'])
        fig_day_records = style_chart(fig_day_records, "sold packages by Day of Week")
        st.plotly_chart(fig_day_records, use_container_width=True)
    with col2:
        st.markdown("<h5>Revenue by Day of Week</h5>", unsafe_allow_html=True)
        fig_day_revenue = px.bar(day_data, x='Day', y='Revenue', labels={'Revenue': 'Revenue (AED)'}, color_discrete_sequence=['#FF6F61'])
        fig_day_revenue = style_chart(fig_day_revenue, "Revenue by Day of Week")
        st.plotly_chart(fig_day_revenue, use_container_width=True)
else:
    st.info("No valid date data to display day of week trends.")

st.markdown("<h5>sold packages by Hour of Day</h5>", unsafe_allow_html=True)
try:
    output_df['TimeParsed'] = pd.to_datetime(output_df['Time'], format='%H:%M', errors='coerce')
    valid_times_df = output_df[output_df['TimeParsed'].notna()]
    if not valid_times_df.empty:
        time_data_hr = valid_times_df.groupby(valid_times_df['TimeParsed'].dt.hour)['No'].count().reset_index()
        time_data_hr.columns = ['Hour', 'Records']
        all_hours = pd.DataFrame({'Hour': range(24)})
        time_data_hr = all_hours.merge(time_data_hr, on='Hour', how='left').fillna({'Records': 0})
        time_data_hr['HourLabel'] = time_data_hr['Hour'].apply(lambda x: f"{x:02d}:00")
        
        fig_time_records = px.line(
            time_data_hr, x='HourLabel', y='Records',
            labels={'Records': 'Number of sold packages', 'HourLabel': 'Hour of Day'}, color_discrete_sequence=['#FF6F61']
        )
        fig_time_records = style_chart(fig_time_records, "sold packages by Hour of Day")
        st.plotly_chart(fig_time_records, use_container_width=True)
    else:
        st.info("No valid time data to display hour of day trends.")
except KeyError:
    st.info("Hour of Day analysis unavailable ('Time' column is missing).")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Discount Insights Section ----------
st.markdown("<h3>Discount Insights</h3>", unsafe_allow_html=True)
total_records_for_discount = len(output_df)
discount_records = output_df['Desc Code'].dropna().loc[output_df['Desc Code'].str.strip() != ''].count()
discount_percentage = (discount_records / total_records_for_discount * 100) if total_records_for_discount > 0 else 0
top_discounts = output_df['Desc Code'].dropna().loc[output_df['Desc Code'].str.strip() != ''].value_counts().head(5)

col1, col2 = st.columns(2)
with col1:
    st.metric("sold packages with Discount", discount_records)
with col2:
    st.metric("Percentage with Discount", f"{discount_percentage:.2f}%")

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h5>Top 5 Discount Codes by Usage</h5>", unsafe_allow_html=True)
    st.dataframe(top_discounts.map(format_number), use_container_width=True)
with col2:
    st.markdown("<h5>Chart for Top 5 Discount Codes</h5>", unsafe_allow_html=True)
    fig7 = px.bar(
        top_discounts.reset_index(), x='Desc Code', y='count',
        labels={'count': 'Number of Uses'}, color_discrete_sequence=['#FF6F61']
    )
    fig7 = style_chart(fig7, "Top 5 Discount Codes by Usage")
    fig7.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig7, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Heatmap Section ----------
st.markdown("<h3>Heatmap (Package vs. Location)</h3>", unsafe_allow_html=True)
heatmap_data = pivot_df_filtered.drop('Total', axis=0, errors='ignore')[['CITY WALK', 'INDEX', 'DKP']]
heatmap_data_normalized = heatmap_data.div(heatmap_data.max(axis=1), axis=0).fillna(0)
heatmap_text = heatmap_data.astype(str)
fig8 = px.imshow(
    heatmap_data_normalized,
    labels=dict(x="Location", y="Package", color="Normalized Count"),
    color_continuous_scale=['#6B7280', '#00899D', '#FF6F61'],
    text_auto=False, aspect='auto'
)
# Apply the base style
fig8 = style_chart(fig8, "Booking Counts Matrix (Package vs. Location)")

# Update traces and layout with specific adjustments for this chart
fig8.update_traces(text=heatmap_text.values, texttemplate='%{text}', textfont=dict(size=12, color='white'))
fig8.update_layout(
    height=800,  # Increased height for more vertical space
    margin=dict(t=100, b=50, l=50, r=50), # Increased top margin to move title up
    font_size=12, 
    xaxis_nticks=len(heatmap_data.columns), 
    yaxis_nticks=len(heatmap_data.index)
)
st.plotly_chart(fig8, use_container_width=True)


# --- New Deeper Insights Sections ---

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("<h2>Deeper Insights & Advanced Analytics</h2>", unsafe_allow_html=True)

# ---------- Customer Demographics Analysis Section ----------
st.markdown("<h3>Customer Demographics Analysis</h3>", unsafe_allow_html=True)

# Check if the raw dataframe from the main page is available
if 'raw_df' in st.session_state and st.session_state.raw_df is not None:
    raw_df = st.session_state.raw_df
    # Filter raw_df to be consistent with the main analysis
    raw_df = raw_df[raw_df['servicename'].str.upper() != 'UAE NATIONAL PRE-EMPLOYMENT']

    # --- Create a clean base for demographic analysis ---
    # Drop duplicates based on patientname to analyze unique customers first
    unique_customers_df = raw_df.copy().drop_duplicates(subset=['patientname'], keep='first')

    # --- Age Analysis on Unique Customers ---
    st.markdown("<h5>Analysis by Age Group</h5>", unsafe_allow_html=True)
    if 'age' in unique_customers_df.columns and 'netamt' in unique_customers_df.columns:
        # Clean the age data from the unique customers dataframe
        age_df = unique_customers_df[['age', 'netamt']].copy()
        
        # --- THIS IS THE FIX ---
        # Convert age to string and remove non-numeric characters like 'y'
        age_df['age'] = age_df['age'].astype(str).str.replace(r'\D', '', regex=True)
        # -----------------------
        
        age_df['age'] = pd.to_numeric(age_df['age'], errors='coerce')
        age_df.dropna(subset=['age'], inplace=True)
        
        if not age_df.empty:
            age_df['age'] = age_df['age'].astype(int)

            # Define age bins and labels
            bins = [0, 20, 30, 40, 50, 60, 100]
            labels = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60+']
            age_df['Age Group'] = pd.cut(age_df['age'], bins=bins, labels=labels, right=False)

            # Group by the newly created age groups
            age_analysis = age_df.groupby('Age Group', observed=False).agg(
                Count=('age', 'count'),
                Total_Revenue=('netamt', 'sum')
            ).reset_index()

            col1, col2 = st.columns(2)
            with col1:
                # --- Gender Analysis on Unique Customers ---
                if 'gender' in unique_customers_df.columns:
                    gender_df = unique_customers_df[['gender']].copy()
                    gender_df.dropna(subset=['gender'], inplace=True)
                    
                    if not gender_df.empty:
                        gender_counts = gender_df['gender'].value_counts().reset_index()
                        
                        fig_gender = px.pie(
                            gender_counts, 
                            names='gender', 
                            values='count',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_gender = style_chart(fig_gender, "Gender Distribution of Unique Customers")
                        st.plotly_chart(fig_gender, use_container_width=True)
                    else:
                        st.info("No valid gender data found for unique customers.")
                else:
                    st.info("Gender analysis is unavailable. Ensure the 'gender' column exists in the uploaded file.")
            with col2:
                fig_age_revenue = px.bar(age_analysis, x='Age Group', y='Total_Revenue', labels={'Total_Revenue': 'Total Revenue (AED)'})
                fig_age_revenue = style_chart(fig_age_revenue, "Total Revenue by Age Group")
                st.plotly_chart(fig_age_revenue, use_container_width=True)
        else:
            st.info("No valid age data found for unique customers.")
    else:
        st.info("Age analysis is unavailable. Ensure 'age' and 'netamt' columns exist in the uploaded file.")

else:
    st.warning("Raw data not found. Please re-upload the file on the main page to enable this analysis.")

# 1. Advanced Customer Segmentation
st.markdown("<h3>Customer Segmentation: Identifying VIP Customers</h3>", unsafe_allow_html=True)
if not output_df.empty:
    rfm_df = output_df.groupby('Name').agg(
        Frequency=('Date', 'count'),
        Monetary=('Price', 'sum')
    ).sort_values(by='Monetary', ascending=False).reset_index()
    st.markdown("<h5>Top 10 Customers by Total Spending</h5>", unsafe_allow_html=True)
    st.dataframe(rfm_df.head(10), use_container_width=True, hide_index=True)
else:
    st.info("Not enough data for customer segmentation.")

# 2. Scenario & Predictive Analysis
st.markdown("<h3>Revenue Trendline</h3>", unsafe_allow_html=True)
if not time_data.empty and len(time_data) > 1:
    st.markdown("<h5>This chart shows the actual daily revenue as points, with a 7-day moving average trendline to smooth out daily fluctuations and show the overall direction.</h5>", unsafe_allow_html=True)
    try:
        # Calculate a 7-day moving average for the revenue
        # The window size can be adjusted (e.g., 3, 7, 14 days)
        window_size = 7
        if len(time_data) >= window_size:
            time_data['Moving_Avg'] = time_data['Revenue'].rolling(window=window_size).mean()
        else:
            # Handle cases with fewer data points than the window size
            time_data['Moving_Avg'] = time_data['Revenue'].rolling(window=len(time_data)).mean()

        # Create a figure with Plotly Graph Objects to combine traces
        fig_trend = go.Figure()

        # Add the scatter plot for actual daily revenue
        fig_trend.add_trace(go.Scatter(
            x=time_data['Date'],
            y=time_data['Revenue'],
            mode='markers',
            name='Daily Revenue',
            marker=dict(color='rgba(0, 137, 157, 0.6)') # Use a semi-transparent theme color
        ))

        # Add the line plot for the moving average
        fig_trend.add_trace(go.Scatter(
            x=time_data['Date'],
            y=time_data['Moving_Avg'],
            mode='lines',
            name=f'{window_size}-Day Moving Average',
            line=dict(color='yellow', width=3) # Highlight the trendline
        ))
        
        # Style the chart to match the dashboard
        fig_trend = style_chart(fig_trend, "Overall Revenue Trend (7-Day Moving Average)")
        
        # Display the chart
        st.plotly_chart(fig_trend, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while creating the trendline chart: {e}")
else:
    st.info("Not enough data for predictive analysis (at least 2 data points required).")

# 4. Advanced Discount Impact Analysis
st.markdown("<h3>Advanced Discount Impact Analysis</h3>", unsafe_allow_html=True)
output_df['HasDiscount'] = output_df['Desc Code'].fillna('').str.strip().ne('')
discounted_df = output_df[output_df['HasDiscount']]
nondiscounted_df = output_df[~output_df['HasDiscount']]
avg_discounted_value = discounted_df['Price'].mean() if not discounted_df.empty else 0
avg_nondiscounted_value = nondiscounted_df['Price'].mean() if not nondiscounted_df.empty else 0

col1, col2 = st.columns(2)
with col1:
    st.metric("Avg. Invoice (with Discount)", f"{avg_discounted_value:,.0f} AED")
with col2:
    st.metric("Avg. Invoice (without Discount)", f"{avg_nondiscounted_value:,.0f} AED")

if not discounted_df.empty:
    discount_revenue = discounted_df.groupby('Desc Code')['Price'].agg(['count', 'sum']).rename(columns={'count': 'Usage Count', 'sum': 'Total Revenue'}).sort_values(by='Total Revenue', ascending=False).head(10)
    st.markdown("<h5>Top 10 Discount Codes by Revenue</h5>", unsafe_allow_html=True)
    st.dataframe(discount_revenue.style.format({'Total Revenue': '{:,.0f} AED'}), use_container_width=True)
else:
    st.info("No discount data available for this report.")

# 3. Market Basket Analysis
st.markdown("<h3>Market Basket Analysis (Most Co-purchased Packages)</h3>", unsafe_allow_html=True)
daily_purchases = output_df.groupby(['Name', output_df['Date'].dt.date])['Package'].apply(list).reset_index()
multi_package_purchases = daily_purchases[daily_purchases['Package'].apply(len) > 1]
package_pairs = {}
for index, row in multi_package_purchases.iterrows():
    packages = sorted(list(set(row['Package'])))
    for pair in combinations(packages, 2):
        package_pairs[pair] = package_pairs.get(pair, 0) + 1

top_pairs = pd.DataFrame() # Initialize empty dataframe
if package_pairs:
    top_pairs = pd.DataFrame(package_pairs.items(), columns=['Packages', 'Co-purchase Count'])
    top_pairs = top_pairs.sort_values(by='Co-purchase Count', ascending=False).head(10)
    top_pairs['Packages_str'] = top_pairs['Packages'].apply(lambda x: f"{x[0]} + {x[1]}")
    st.dataframe(top_pairs[['Packages_str', 'Co-purchase Count']].rename(columns={'Packages_str': 'Packages'}), use_container_width=True, hide_index=True)
else:
    st.info("Not enough data to show co-purchased packages.")

# 5. Interactive Co-purchase Network
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("<h3>Interactive Co-purchase Network</h3>", unsafe_allow_html=True)
st.markdown("<h6>Click the package's node to highlight the links</h6>", unsafe_allow_html=True)

if not top_pairs.empty:
    # Create edges from the top_pairs dataframe
    edges = []
    for index, row in top_pairs.iterrows():
        src, dst = row['Packages']
        weight = row['Co-purchase Count']
        edges.append((src, dst, weight))

    # Create network with enhanced styling to match the dashboard
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1E1E1E",  # Dark background to match chart containers
        font_color="#FFFFFF"   # White font for readability
    )

    # Enhanced physics for better node distribution
    net.barnes_hut(
        gravity=-50000,
        central_gravity=0.15,
        spring_length=400,
        spring_strength=0.01
    )

    # Calculate maximum weight for edge scaling
    max_weight = max(w for _, _, w in edges) if edges else 1

    # Add nodes and edges
    for src, dst, weight in edges:
        line_width = (weight / max_weight) * 15 if max_weight > 0 else 1
        for node in (src, dst):
            if node not in [n["id"] for n in net.nodes]:
                net.add_node(
                    node, label=node,
                    color={
                        "border": "#00c2d1",        # Bright accent color from theme
                        "background": "#005f6b",    # Darker shade of theme color
                        "highlight": {
                            "border": "#FFFFFF",    # White border on highlight
                            "background": "#00899D" # Main theme color on highlight
                        }
                    },
                    font={"size": 22, "face": "Roboto", "color": "#FFFFFF"} # Increased font size
                )
        net.add_edge(
            src, 
            dst, 
            title=f"{src} ↔ {dst}<br>Co-purchase Count: {weight}", 
            width=line_width, 
            color="#6b7280", 
            originalWidth=line_width
        )

    # Define network options as a JSON string
    options = """
    {
      "interaction": { "hover": true, "navigationButtons": true, "multiselect": true },
      "edges": { 
        "color": { "inherit": false, "highlight": "#00c2d1" }, 
        "smooth": { "type": "continuous" } 
      },
      "nodes": { 
        "shape": "dot", 
        "size": 25, 
        "font": { "size": 22, "face": "Roboto", "color": "#FFFFFF" }, 
        "borderWidth": 2, 
        "shadow": { "enabled": true, "size": 10, "x": 5, "y": 5 } 
      },
      "physics": { 
        "barnesHut": { "gravitationalConstant": -50000, "centralGravity": 0.15, "springLength": 400, "springConstant": 0.01 }, 
        "minVelocity": 0.1, 
        "solver": "barnesHut" 
      }
    }
    """
    
    # Custom JavaScript for dynamic edge highlighting
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        if (typeof network !== 'undefined') {
            network.on('selectNode', function(params) {
                var selectedNode = params.nodes[0];
                var edges = network.body.data.edges;
                edges.forEach(function(edge) {
                    var baseWidth = edge.originalWidth || edge.width;
                    if (edge.from === selectedNode || edge.to === selectedNode) {
                        edges.update({id: edge.id, color: '#00c2d1', width: baseWidth * 1.5});
                    } else {
                        edges.update({id: edge.id, color: '#6b7280', width: baseWidth});
                    }
                });
            });

            network.on('deselectNode', function(params) {
                var edges = network.body.data.edges;
                edges.forEach(function(edge) {
                    var baseWidth = edge.originalWidth || edge.width;
                    edges.update({id: edge.id, color: '#6b7280', width: baseWidth});
                });
            });
        }
    });
    </script>
    """

    # Save the network to HTML and render
    try:
        net.save_graph("network.html")
        with open("network.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Inject options and custom JS
        options_injection = f'network.setOptions({options});'
        html_content = re.sub(r'(var network = new vis\.Network\(container, data, .*?\);)', r'\\1\\n' + options_injection, html_content)
        html_content = html_content.replace('</body>', f'{custom_js}</body>')
        
        components.html(html_content, height=750, scrolling=True)
    except Exception as e:
        st.error(f"Error rendering network: {str(e)}")
else:
    st.info("Not enough co-purchase data to generate a network graph.")

# ---------- Footer ----------
load_footer()
