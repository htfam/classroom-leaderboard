import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime
import pytz
import io
import gspread  # Direct gspread connection

# --- Page Configuration ---
st.set_page_config(
    page_title="Class Competition Leaderboard",
    page_icon="🏆",
    layout="centered"
)

# --- App Title and Description ---
st.title("🏆 Data Competition Leaderboard")
st.markdown("""
Welcome to the (optional) midterm data competition! Submit your predictions (good/bad) to see how you rank against your peers. First place gets 5 points added to their midterm exam grade, second places gets 3 points, and third place gets 1 point.
The evaluation metric is **F1 Score**. Higher is better!

<u>Note: You can upload one or multiple CSV files at once to test different models. All **15,696 rows** matching the sample submission file must be submitted. Missing or unmatched predictions will be flagged as invalid.</u>
""", unsafe_allow_html=True)

# --- Session State for File Uploader Reset ---
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def clear_file_uploader():
    st.session_state["uploader_key"] += 1

# --- Direct Gspread Connection ---
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


# --- Helper Functions ---
@st.cache_data(ttl=60)
def fetch_leaderboard():
    """Fetches submission data and builds leaderboard dataframes."""
    try:
        values = worksheet.get_all_values()
        
        # Return empty table if sheet has no data rows
        if not values or len(values) <= 1:
            return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp', 'File Name'])

        # Read data rows (skipping header row 0)
        data_rows = values[1:]
        df = pd.DataFrame(data_rows)

        # Ensure at least 4 columns exist (0: Name, 1: Score, 2: Timestamp, 3: File Name)
        for col_idx in range(4):
            if col_idx not in df.columns:
                df[col_idx] = "N/A"

        # Select first 4 columns and assign explicit header names
        df = df.iloc[:, :4]
        df.columns = ['Name', 'Score', 'Timestamp', 'File Name']

        # Clean file names and missing strings
        df['File Name'] = df['File Name'].replace('', 'N/A').fillna('N/A')

        # Clean and convert numeric scores
        df = df[df['Score'] != '']
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df.dropna(subset=['Score'], inplace=True)

        # Calculate Score Rank across all submissions
        df_sorted_score = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        df_sorted_score['Rank'] = df_sorted_score.index + 1

        # Convert timestamp to datetime for reliable chronological sorting
        df_sorted_score['dt'] = pd.to_datetime(
            df_sorted_score['Timestamp'].str.rsplit(' ', n=1).str[0], 
            errors='coerce'
        )

        # Sort all submissions by most recent timestamp first
        df_most_recent = df_sorted_score.sort_values(by="dt", ascending=False).reset_index(drop=True)

        return df_most_recent[['Rank', 'Name', 'Score', 'Timestamp', 'File Name']]
    except Exception as e:
        st.error(f"An error occurred while reading the leaderboard: {e}")
        return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp', 'File Name'])


def calculate_f1_score(submission_df, solution_df):
    """Calculates F1 Score after validating and merging submission and solution files."""
    
    # 1. Flexible Column Identification
    id_col = None
    pred_col = None
    for col in submission_df.columns:
        col_lower = str(col).strip().lower()
        if col_lower in ['unique_id', 'id']:
            id_col = col
        elif col_lower in ['prediction', 'label', 'target', 'pred']:
            pred_col = col

    if not id_col or not pred_col:
        raise ValueError("Submission file must contain 'unique_id' and 'prediction' columns.")

    # 2. Check for exact number of rows (15,696)
    if len(submission_df) != len(solution_df):
        raise ValueError(f"Incorrect number of rows. Submission has {len(submission_df)} rows, but must have {len(solution_df)} rows.")

    # 3. Check for exact set of unique_ids
    solution_ids = set(solution_df['unique_id'])
    submission_ids = set(submission_df[id_col])

    if solution_ids != submission_ids:
        missing_ids = solution_ids - submission_ids
        extra_ids = submission_ids - solution_ids
        
        error_messages = []
        if missing_ids:
            error_messages.append(f"missing {len(missing_ids)} required unique_id(s)")
        if extra_ids:
            error_messages.append(f"contains {len(extra_ids)} unexpected unique_id(s)")
            
        raise ValueError(f"Submission file has invalid IDs: {', '.join(error_messages)}.")

    # 4. Clean & Normalize Predictions
    sub_cleaned = submission_df[[id_col, pred_col]].copy()
    sub_cleaned.columns = ['unique_id', 'raw_prediction']
    
    if sub_cleaned['raw_prediction'].isnull().any():
        raise ValueError("Submission contains missing (NaN) prediction values.")

    # Convert inputs ('good'/'bad', 0/1, string/numeric) into normalized labels
    def normalize_labels(s):
        s_str = s.astype(str).str.strip().str.lower()
        mapping = {
            'good': 'good', 'bad': 'bad',
            '0': 'good', '1': 'bad',
            '0.0': 'good', '1.0': 'bad',
            'false': 'good', 'true': 'bad'
        }
        return s_str.map(mapping)

    sub_cleaned['submission_target'] = normalize_labels(sub_cleaned['raw_prediction'])
    if sub_cleaned['submission_target'].isnull().any():
        raise ValueError("Invalid predictions found. Values must be 'good', 'bad', 0, or 1.")

    sol_cleaned = solution_df[['unique_id', 'label']].copy()
    sol_cleaned['solution_target'] = normalize_labels(sol_cleaned['label'])

    # 5. Merge on unique_id
    merged_df = pd.merge(sub_cleaned, sol_cleaned, on='unique_id', how='inner')

    # 6. Calculate F1 Score (weighted average for class imbalance)
    score = f1_score(merged_df['solution_target'], merged_df['submission_target'], average='weighted')
    return float(score)


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
    st.header("📥 Make Submissions")
    team_name = st.text_input("Enter your Name", key="team_name")
    
    # File uploader with dynamic key for reset functionality
    uploaded_files = st.file_uploader(
        "Upload your submission CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state['uploader_key']}",
        help="The file(s) must have two columns: 'unique_id' and 'prediction' containing all 15,696 rows."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.button("Submit Predictions", use_container_width=True)
    with col2:
        st.button("Clear Uploads", on_click=clear_file_uploader, use_container_width=True)

    st.markdown("---")
    st.header("📚 Resources")
    try:
        with open("submission_template.csv", "rb") as f:
            st.download_button(
                label="Download Submission Template",
                data=f,
                file_name="submission_template.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.warning("`submission_template.csv` sample template not found in repository.")


# --- Submission Logic ---
if submit_button:
    if not team_name.strip():
        st.sidebar.warning("Please enter your name.")
    elif not uploaded_files:
        st.sidebar.warning("Please upload at least one submission file.")
    else:
        new_rows = []
        submission_results = []

        with st.spinner(f"Scoring {len(uploaded_files)} file(s)..."):
            for file in uploaded_files:
                try:
                    submission_df = pd.read_csv(file)
                    score = calculate_f1_score(submission_df, solution_df)
                    timestamp = datetime.now(pytz.timezone("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z")

                    # Record format: Name | Score | Timestamp | File Name
                    new_rows.append([team_name.strip(), score, timestamp, file.name])
                    submission_results.append((file.name, score, None))
                except Exception as e:
                    submission_results.append((file.name, None, str(e)))

        # Append all valid submissions in one batch request to Google Sheets
        if new_rows:
            try:
                worksheet.append_rows(new_rows, value_input_option='USER_ENTERED')
                st.cache_data.clear() # Refresh leaderboard cache immediately
            except Exception as e:
                st.sidebar.error(f"Failed to update Google Sheets: {e}")

        # Display results for each file in the sidebar
        st.sidebar.markdown("### Submission Results")
        for filename, score, err in submission_results:
            if score is not None:
                st.sidebar.success(f"🎉 **{filename}**\n\nYour F1 Score: **{score:.5f}**")
            else:
                st.sidebar.error(f"❌ **{filename}**\n\nValidation Error: {err}")


# --- Display Leaderboard ---
st.header("📊 Live Leaderboard")

all_submissions_df = fetch_leaderboard()

if all_submissions_df.empty:
    st.info("The leaderboard is currently empty. Be the first to make a submission!")
else:
    tab1, tab2 = st.tabs(["All Submissions", "Best Score per Person"])

    with tab1:
        st.markdown("This view shows every single submission made (most recent submission first).")
        st.dataframe(
            all_submissions_df[['Rank', 'Name', 'Score', 'Timestamp', 'File Name']],
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.markdown("This view shows only the highest score for each unique participant.")
        best_scores_df = all_submissions_df.loc[all_submissions_df.groupby('Name')['Score'].idxmax()]
        best_scores_df = best_scores_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        best_scores_df['Rank'] = best_scores_df.index + 1
        
        st.dataframe(
            best_scores_df[['Rank', 'Name', 'Score', 'Timestamp', 'File Name']],
            use_container_width=True,
            hide_index=True
        )

if st.button('Refresh Leaderboard'):
    st.cache_data.clear()
    st.rerun()
