import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pytz
import io
# We are importing the actual library to catch its specific errors
from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound

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
The evaluation metric is **Root Mean Squared Error (RMSE)**. Lower is better!
""")

# --- NEW: Enhanced Google Sheets Connection with Detailed Error Handling ---
try:
    conn = st.connection("gsheets", type="legacy_gsheets")
except Exception as e:
    # This will catch errors in the initial credential setup
    st.error("Failed to create the initial connection. This usually means a problem with your secrets.toml format.")
    st.error(f"**Detailed Error:** {e}")
    st.stop()


# --- Helper Functions ---
@st.cache_data(ttl=60)
def fetch_leaderboard():
    """Fetches and sorts the leaderboard from Google Sheets with detailed error handling."""
    try:
        df = conn.read(worksheet="leaderboard", usecols=list(range(3)), header=0)
        df.dropna(subset=['Score'], inplace=True)
        df['Score'] = pd.to_numeric(df['Score'])
        df_sorted = df.sort_values(by="Score", ascending=True).reset_index(drop=True)
        df_sorted['Rank'] = df_sorted.index + 1
        return df_sorted[['Rank', 'Name', 'Score', 'Timestamp']]
    # --- NEW: Catching specific, common errors from the gspread library ---
    except SpreadsheetNotFound:
        st.error("Connection successful, but the spreadsheet was not found. Please double-check the `spreadsheet` URL in your secrets.")
        return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])
    except WorksheetNotFound:
        st.error("Spreadsheet was found, but the worksheet/tab named 'leaderboard' was not. Please ensure the tab name is correct (all lowercase).")
        return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])
    except Exception as e:
        # This will catch any other errors during the read process
        st.error("An error occurred while trying to read data from the sheet.")
        st.error(f"**Detailed Error:** {e}")
        return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])

def calculate_rmse(submission_df, solution_df):
    """Calculates RMSE after merging submission and solution files."""
    merged_df = pd.merge(submission_df, solution_df, on='ID', how='left')
    if merged_df['Target_y'].isnull().any():
        raise ValueError("Submission file is missing some IDs present in the solution.")
    rmse = np.sqrt(mean_squared_error(merged_df['Target_y'], merged_df['Target_x']))
    return rmse

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

# --- Submission Logic ---
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
                score = calculate_rmse(submission_df, solution_df)

            timestamp = datetime.now(pytz.timezone("UTC")).strftime("%Y-%m-%d %H:%M:%S UTC")
            new_entry = pd.DataFrame([[team_name, score, timestamp]], columns=["Name", "Score", "Timestamp"])
            
            # Use the underlying gspread worksheet object to append
            worksheet = conn.get_worksheet(0) # 0 is the first sheet
            worksheet.append_rows(new_entry.values.tolist(), value_input_option='USER_ENTERED')

            st.sidebar.success(f"🎉 Submission successful!\n\nYour RMSE score: **{score:.5f}**")
            st.cache_data.clear()
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

if st.button('Refresh Leaderboard'):
    st.cache_data.clear()
    st.rerun()

