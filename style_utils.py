import streamlit as st

def load_css():
    """
    Loads all the shared CSS for the application.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        /* --- General App Styling --- */
        .stApp {
            background-image: url("https://analytics.smartsalem.tech/smartsuite/bg-login.688761c8fcdbbeb1.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            min-height: 100vh;
        }
         header[data-testid="stHeader"] {
            display: none;
        }
                /* Hide bottom right profile menu */
        button[aria-label="User menu"], 
        div[data-testid="stToolbar"] {
            display: none !important;
        }
                /* Hide the entire sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }
           /* Hide profile menu and other floating buttons bottom-right */
        .css-1v3fvcr.e1fqkh3o3,  /* Common profile button container class, may vary */
        [data-testid="stToolbar"],
        button[aria-label="User menu"],
        div[role="complementary"] {
            display: none !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
        }
        /* Expand main content to full width */
        [data-testid="stAppViewContainer"] {
            margin-left: 0;
        }
        /* --- Text and Headers --- */
        h1, h2, h3, h4, h5, .stMarkdown, .stMarkdown div, .stMarkdown p, .stMetric label, .stMetric div {
            color: white !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
            font-family: 'Roboto', sans-serif !important;
        }

        /* --- Main Page Specifics (app.py) --- */
        .stFileUploader label {
            color: white !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        }
        [data-testid="stFileUploader"] .uploadedFileName,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div {
            color: white !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        }
        .download-button {
            background-color: #28a745 !important;
            color: white !important;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            margin-right: 10px;
        }
        .analytics-button {
            background-color: #007bff !important;
            color: white !important;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }

        /* --- Analytics Page Specifics (analyticss.py) --- */
        div[data-testid="stMetric"] {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            text-align: center !important;
            min-height: 140px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            margin-bottom: 10px !important;
        }
        div[data-testid="stMetric"] label {
            font-weight: bold !important;
            font-size: 1.2em !important;
        }
        div[data-testid="stMetric"] div {
            font-size: 1.5em !important;
            font-weight: 700;
        }
        .section-divider {
            border-top: 2px solid #00899D;
            margin: 40px 0;
        }
        .stDataFrame, .stTable {
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def load_footer():
    """
    Loads the shared footer for the application.
    """
    st.markdown("""
        <div class="section-divider"></div>
        <br>
    """, unsafe_allow_html=True)
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
