import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime
import pytz
import io
import gspread  # Direct gspread connection as used in your previous deployment

# --- Page Configuration ---
st.set_page_config(
    page_title="Class Competition Leaderboard",
    page_icon="🏆",
    layout="centered"
)

# --- App Title and Description ---
st.title("🏆 Data Competition Leaderboard")
st.markdown("""
Welcome to the (optional) midterm data competition! Submit your predictions to see how you rank against your peers. First place gets 5 points added to their midterm exam grade, second places gets 3 points, and third place gets 1 point.
The evaluation metric is **F1 Score**. Higher is better!

<u>Note: All **15,696 rows** matching the sample submission file must be submitted. Missing or unmatched predictions will be flagged as invalid.</u>
""", unsafe_allow_html=True)

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
    """Fetches and sorts the leaderboard from the Google Sheet."""
    try:
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        
        if df.empty:
            return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])

        df.dropna(subset=['Score'], inplace=True)
        df['Score'] = pd.to_numeric(df['Score'])
        # Sort by score descending (higher F1 score is better)
        df_sorted = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        df_sorted['Rank'] = df_sorted.index + 1
        return df_sorted[['Rank', 'Name', 'Score', 'Timestamp']]
    except Exception as e:
        st.error(f"An error occurred while reading the leaderboard: {e}")
        return pd.DataFrame(columns=['Rank', 'Name', 'Score', 'Timestamp'])


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
    st.header("📥 Make a Submission")
    team_name = st.text_input("Enter your Name or Team Name", key="team_name")
    uploaded_file = st.file_uploader(
        "Upload your submission CSV file",
        type=["csv"],
        help="The file must have two columns: 'unique_id' and 'prediction' containing all 15,696 rows."
    )
    submit_button = st.button("Submit Predictions")
    st.markdown("---")
    st.header("📚 Resources")
    try:
        with open("submission.csv", "rb") as f:
            st.download_button(
                label="Download Sample Submission",
                data=f,
                file_name="sample_submission.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        #st.warning("`submission.csv` sample template not found in repository.")


# --- Submission Logic ---
if submit_button:
    if not team_name.strip():
        st.sidebar.warning("Please enter your name or team name.")
    elif uploaded_file is None:
        st.sidebar.warning("Please upload your submission file.")
    else:
        try:
            submission_df = pd.read_csv(uploaded_file)

            with st.spinner("Scoring your submission..."):
                score = calculate_f1_score(submission_df, solution_df)

            timestamp = datetime.now(pytz.timezone("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z")
            new_entry = pd.DataFrame([[team_name.strip(), score, timestamp]], columns=["Name", "Score", "Timestamp"])
            
            # Append the new row to Google Sheets via gspread
            worksheet.append_rows(new_entry.values.tolist(), value_input_option='USER_ENTERED')

            st.sidebar.success(f"🎉 Submission successful!\n\nYour F1 Score: **{score:.5f}**")
            st.cache_data.clear() # Clear cache to display update immediately
        except ValueError as ve:
            st.sidebar.error(f"Validation Error: {ve}")
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")


# --- Display Leaderboard ---
st.header("📊 Live Leaderboard")

all_submissions_df = fetch_leaderboard()

if all_submissions_df.empty:
    st.info("The leaderboard is currently empty. Be the first to make a submission!")
else:
    tab1, tab2 = st.tabs(["All Submissions", "Best Score per Person"])

    with tab1:
        st.markdown("This view shows every single submission made.")
        st.dataframe(
            all_submissions_df,
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.markdown("This view shows only the highest score for each unique participant.")
        best_scores_df = all_submissions_df.loc[all_submissions_df.groupby('Name')['Score'].idxmax()]
        best_scores_df = best_scores_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        best_scores_df['Rank'] = best_scores_df.index + 1
        
        st.dataframe(
            best_scores_df[['Rank', 'Name', 'Score', 'Timestamp']],
            use_container_width=True,
            hide_index=True
        )

if st.button('Refresh Leaderboard'):
    st.cache_data.clear()
    st.rerun()
