import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score  # Changed from mean_squared_error
from datetime import datetime
import pytz
import io
import gspread  # Import the gspread library directly

# --- Page Configuration ---
st.set_page_config(
    page_title="Class Competition Leaderboard",
    page_icon="🏆",
    layout="centered"
)

# --- App Title and Description ---
st.title("🏆 Data Competition Leaderboard")
st.markdown("""
Welcome to the class data competition! Submit your predictions to see how you rank against your peers.
The evaluation metric is **F1 Score**. Higher is better!
""")

# --- FINAL: Direct Gspread Connection (Bypassing st.connection) ---
try:
    # Use st.secrets to get credentials for gspread
    creds = st.secrets["connections"]["gsheets"]
    client = gspread.service_account_from_dict(creds)

    # Open the spreadsheet using the URL from secrets
    spreadsheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
    spreadsheet = client.open_by_url(spreadsheet_url)
    worksheet = spreadsheet.worksheet("leaderboard")
except Exception as e:
    st.error("Failed to connect to Google Sheets. Please double-check all your secrets and sharing settings.")
    st.error(f"**Detailed Error:** {e}")
    st.stop()


# --- Helper Functions (Updated to use the new worksheet object) ---
@st.cache_data(ttl=60)
def fetch_leaderboard():
    """Fetches and sorts the leaderboard from the Google Sheet."""
    try:
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        
        # If the sheet is empty, return a blank dataframe
        if df.empty:
            return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])

        df.dropna(subset=['Score'], inplace=True)
        df['Score'] = pd.to_numeric(df['Score'])
        # Sort by score (descending, since higher F1 score is better)
        df_sorted = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        df_sorted['Rank'] = df_sorted.index + 1
        return df_sorted[['Rank', 'Name', 'Score', 'Timestamp']]
    except Exception as e:
        st.error(f"An error occurred while reading the leaderboard: {e}")
        return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])

def calculate_f1_score(submission_df, solution_df):
    """Calculates F1 Score after merging submission and solution files."""
    merged_df = pd.merge(submission_df, solution_df, on='ID', how='left')
    if merged_df['Target_y'].isnull().any():
        raise ValueError("Submission file is missing some IDs present in the solution.")
    # Calculate F1 Score (using 'weighted' for multiclass/imbalanced datasets)
    score = f1_score(merged_df['Target_y'], merged_df['Target_x'], average='weighted')
    return score

# --- Load Solution File from Secrets ---
try:
    csv_string = st.secrets["solution_data"]["csv_data"]
    solution_df = pd.read_csv(io.StringIO(csv_string))
except KeyError:
    st.error("Solution data not found in secrets. Please check the `[solution_data]` section of your secrets.")
    st.stop()
except Exception as e:
    st.error(f"Could not parse the solution data from secrets. Error: {e}")
    st.stop()

# --- Sidebar for Submission ---
with st.sidebar:
    st.header("📥 Make a Submission")
    team_name = st.text_input("Enter your Name or Team Name", key="team_name")
    uploaded_file = st.file_uploader(
        "Upload your submission CSV file",
        type=["csv"],
        help="The file must have two columns: 'ID' and 'Target'."
    )
    submit_button = st.button("Submit Predictions")
    st.markdown("---")
    # This resource is no longer needed as there is no sample file in this version
    # You can add it back if you add a sample_submission.csv to your repo


# --- Submission Logic (Updated to use the new worksheet object) ---
if submit_button:
    if not team_name:
        st.sidebar.warning("Please enter your name or team name.")
    elif uploaded_file is None:
        st.sidebar.warning("Please upload your submission file.")
    else:
        try:
            submission_df = pd.read_csv(uploaded_file)
            if not {'ID', 'Target'}.issubset(submission_df.columns):
                raise ValueError("Submission file must contain 'ID' and 'Target' columns.")

            submission_df.columns = ['ID', 'Target_x']
            solution_df.columns = ['ID', 'Target_y']

            with st.spinner("Scoring your submission..."):
                score = calculate_f1_score(submission_df, solution_df)

            timestamp = datetime.now(pytz.timezone("UTC")).strftime("%Y-%m-%d %H:%M:%S UTC")
            new_entry = pd.DataFrame([[team_name, score, timestamp]], columns=["Name", "Score", "Timestamp"])
            
            # Append the new row using the worksheet object we created at the start
            worksheet.append_rows(new_entry.values.tolist(), value_input_option='USER_ENTERED')

            st.sidebar.success(f"🎉 Submission successful!\n\nYour F1 Score: **{score:.5f}**")
            st.cache_data.clear() # Clear cache to show new result immediately
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")

# --- Display Leaderboard ---
st.header("📊 Live Leaderboard")
leaderboard_df = fetch_leaderboard()

if not leaderboard_df.empty:
    st.dataframe(
        leaderboard_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("The leaderboard is currently empty. Be the first to make a submission!")

if st.button('Refresh Leaderboard'):
    st.cache_data.clear()
    st.rerun()

