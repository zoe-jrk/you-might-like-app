import streamlit as st
import pandas as pd
import requests

# --- Load recommendation data ---
@st.cache
def load_recommendations():
    df = pd.read_csv("all_user_mood_recommendations.csv")
    return df

# --- Load full user profiles ---
@st.cache
def load_user_profiles():
    df = pd.read_excel("USER STREAMING BEHAVIOR ANALYSIS_cleaned.xlsx", sheet_name=0)
    return df

recommendations = load_recommendations()
users = load_user_profiles()

# --- TMDb Poster Fetch Function with Spider Woman Custom Override ---
def fetch_poster(title):
    api_key = "5cf3bf2689dc822a35ad0ec1dff722f1"  # <<< Replace with your real TMDb API key

    # ✅ Custom poster for Spider Woman
    if "Spider-Woman" in title:
        return "https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FMadame_Web_%2528film%2529&psig=AOvVaw0rhAb9qT2kS14Z_UW6LXZr&ust=1751189754468000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCLieg6_ok44DFQAAAAAdAAAAABAE"  # Example, replace with your desired image URL

    # ✅ TMDb API for all other titles
    query = title.replace(' ', '%20')
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}"
    try:
        response = requests.get(url).json()
        if response['results']:
            poster_path = response['results'][0]['poster_path']
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return "https://via.placeholder.com/200x300?text=No+Image"
    except:
        return "https://via.placeholder.com/200x300?text=No+Image"

# --- CSS Styling for Disney Blue and Dark Mode ---
page_bg = """
<style>
body {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3, h4, h5 {
    color: #1e90ff;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/3/3e/Disney%2B_logo.svg' width='200'>
        <h1>You Might Like</h1>
        <p>Your Personalized Disney+ Experience</p>
        <hr style='border-top: 3px solid #1e90ff;'>
    </div>
""", unsafe_allow_html=True)

# --- User and Mood Selection ---
user_ids = sorted(recommendations['user_id'].unique())
selected_user = st.selectbox("Select User ID", user_ids)

moods = sorted(recommendations[recommendations['user_id'] == selected_user]['mood'].unique())
selected_mood = st.selectbox("Select Mood", moods)

# --- Filter Recommendations ---
filtered = recommendations[
    (recommendations['user_id'] == selected_user) &
    (recommendations['mood'] == selected_mood)
].sort_values(by='rank')

# --- Display Recommended Titles Grid (5 per row) ---
st.subheader("Recommended Titles")
titles = filtered['title'].tolist()
cols = st.columns(5)

for i, title in enumerate(titles):
    with cols[i % 5]:
        st.image(fetch_poster(title), use_container_width=True)  # ✅ fixed
        st.caption(f"**{title}**")

# --- Display Full User Profile Below ---
user_info = users[users['user_id'] == selected_user]
if not user_info.empty:
    user_row = user_info.iloc[0]

    st.markdown(f"""
    <div style='background-color: #1e293b; padding: 15px; border-radius: 10px; margin-top: 30px;'>
        <h4 style='color: #38bdf8;'>Full User Profile</h4>
    """, unsafe_allow_html=True)

    for col in users.columns:
        value = user_row[col]
        st.markdown(f"<p><strong>{col.replace('_', ' ')}:</strong> {value}</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
