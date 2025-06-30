import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- STEP 1: Load Data ----------
movies_df = pd.read_csv("disney_plus_shows.csv")
survey_df = pd.read_excel("USER STREAMING BEHAVIOR ANALYSIS_cleaned.xlsx")

movies_df["movieId"] = movies_df.index
movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

# ---------- STEP 2: Popularity ----------
movies_df["popularity"] = pd.to_numeric(movies_df["imdb_votes"].astype(str).str.replace(',', ''), errors="coerce")
movies_df["popularity"].fillna(0, inplace=True)

# ---------- STEP 3: Clean Survey Data ----------
columns_to_convert = ["franchise_attachment_score", "actor_director_attachment_score", "engagement_level", "rewatch_rate"]
for col in columns_to_convert:
    survey_df[col] = pd.to_numeric(survey_df[col], errors="coerce")
survey_df.fillna(0, inplace=True)

# ---------- STEP 4: Mood-Genre Map ----------
mood_genre_map = {
    "mood_happy": ["Comedy", "Family", "Animation"],
    "mood_sad": ["Drama", "Romance"],
    "mood_bored": ["Action", "Adventure", "Sci-Fi"]
}

# ---------- STEP 5: Build Personalized User-Item Interaction Matrix ----------
interaction_data = []
user_ids = survey_df["user_id"].unique()
movie_ids = movies_df["movieId"].unique()

np.random.seed(42)
for user in user_ids:
    user_row = survey_df[survey_df["user_id"] == user].iloc[0]
    preferred_genres = str(user_row["preferred_genres"]).split(",") if pd.notna(user_row["preferred_genres"]) else []
    favorite_franchise = str(user_row["favourite_franchise"]).lower()
    franchise_attachment = user_row["franchise_attachment_score"]
    actor_director_attachment = user_row["actor_director_attachment_score"]
    engagement_level = user_row["engagement_level"]
    preferred_duration = str(user_row["preferred_duration"])
    social_rec_use = 1 if str(user_row["uses_social_recommendations"]).lower() == "yes" else 0
    media_discovery_use = 1 if str(user_row["uses_media_recommendations"]).lower() == "yes" else 0

    mood_genre_boost = {}
    for mood_col in ["mood_happy", "mood_sad", "mood_bored"]:
        mood_value = pd.to_numeric(user_row[mood_col], errors='coerce')
        if pd.notna(mood_value) and mood_value > 0:
            for g in mood_genre_map.get(mood_col, []):
                mood_genre_boost[g] = mood_genre_boost.get(g, 0) + 1.5

    for movie_id in movie_ids:
        movie_row = movies_df.loc[movies_df["movieId"] == movie_id].iloc[0]
        movie_genres = str(movie_row.get("genre", "")).split("|")
        movie_title_lower = str(movie_row["title"]).lower()

        genre_match = len(set(preferred_genres) & set(movie_genres)) * 2.0
        mood_match = sum([mood_genre_boost.get(g, 0) for g in movie_genres])
        franchise_match = franchise_attachment * 2.0 if favorite_franchise in movie_title_lower else 0
        behavioral_score = actor_director_attachment + engagement_level + social_rec_use + media_discovery_use

        total_score = genre_match + mood_match + franchise_match + behavioral_score
        total_score *= np.random.uniform(0.8, 1.2)
        interaction_data.append({"userId": user, "movieId": movie_id, "rating": max(total_score, 0.1)})

ratings_df = pd.DataFrame(interaction_data)

# ---------- STEP 6: Matrix Factorization Model ----------
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        torch.nn.init.uniform_(self.user_factors.weight, 0, 0.05)
        torch.nn.init.uniform_(self.item_factors.weight, 0, 0.05)

    def forward(self, data):
        users, items = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)

class Loader(Dataset):
    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()
        users = self.ratings.userId.unique()
        movies = self.ratings.movieId.unique()

        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.movieid2idx = {o: i for i, o in enumerate(movies)}
        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}

        self.ratings["userId"] = self.ratings.userId.map(self.userid2idx)
        self.ratings["movieId"] = self.ratings.movieId.map(self.movieid2idx)

        self.x = torch.tensor(self.ratings[["userId", "movieId"]].values)
        self.y = torch.tensor(self.ratings["rating"].values)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.ratings)

train_set = Loader(ratings_df)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

n_users = ratings_df.userId.nunique()
n_items = ratings_df.movieId.nunique()

cuda = torch.cuda.is_available()
model = MatrixFactorization(n_users, n_items, n_factors=20)
if cuda:
    model = model.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining...")
for epoch in range(50):
    losses = []
    for x_batch, y_batch in train_loader:
        if cuda:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs.squeeze(), y_batch.float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f}")

print("\n✅ Training finished.")

# ---------- STEP 7: TF-IDF Content Similarity ----------
tfidf = TfidfVectorizer()
movie_genres_str = movies_df["genre"].fillna("").astype(str).values
tfidf_matrix = tfidf.fit_transform(movie_genres_str)

def get_content_similarity(user_idx):
    user_id = train_set.idx2userid[user_idx]
    user_row = survey_df[survey_df["user_id"] == user_id].iloc[0]
    user_genres_text = str(user_row["preferred_genres"])
    user_vector = tfidf.transform([user_genres_text])
    cos_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return cos_sim

# ---------- STEP 8: Mood-Specific Hybrid Recommendation ----------
def recommend_top_n_mood(user_idx, mood, model, train_set, movies_df, movie_id_to_title, top_n=30, alpha=0.6, beta=0.2, gamma=0.2):
    model.eval()
    with torch.no_grad():
        user_vector = model.user_factors.weight.data[user_idx]
        mf_scores = torch.matmul(model.item_factors.weight.data, user_vector).cpu().numpy()

        movie_internal_ids = np.arange(len(mf_scores))
        movie_ids = [train_set.idx2movieid[i] for i in movie_internal_ids]

        # Popularity
        popularity_scores = []
        for movie_id in movie_ids:
            pop = movies_df.loc[movies_df["movieId"] == movie_id, "popularity"]
            popularity_scores.append(pop.values[0] if not pop.empty else 0)
        popularity_scores = np.array(popularity_scores)
        popularity_norm = (popularity_scores - popularity_scores.min()) / (popularity_scores.max() - popularity_scores.min() + 1e-8)

        # Content Similarity
        full_content_sim_scores = get_content_similarity(user_idx)
        content_sim_scores = np.array([
            full_content_sim_scores[movies_df[movies_df["movieId"] == movie_id].index[0]] if movie_id in movies_df["movieId"].values else 0
            for movie_id in movie_ids
        ])

        # Mood filtering
        mood_genres = mood_genre_map.get(mood, [])
        mood_mask = np.array([
            any(g in str(movies_df.loc[movies_df["movieId"] == movie_id, "genre"].values[0]) for g in mood_genres)
            for movie_id in movie_ids
        ])

        final_scores = alpha * mf_scores + beta * popularity_norm + gamma * content_sim_scores
        final_scores[~mood_mask] = -np.inf

        top_indices = np.argsort(final_scores)[::-1][:top_n]
        recommended_movie_ids = [movie_ids[i] for i in top_indices]

        recommended_titles = []
        for movie_id in recommended_movie_ids:
            title = movie_id_to_title.get(movie_id, f"Unknown Title (movieId={movie_id})")
            if pd.notna(title) and str(title).lower() != "nan":
                recommended_titles.append(str(title))

    return recommended_titles

# ---------- STEP 9: Example - Recommend for One User and One Mood ----------
user_idx = 1  # Change for different user
mood = "mood_bored"  # Change to mood_sad or mood_bored

recommendations = recommend_top_n_mood(user_idx, mood, model, train_set, movies_df, movie_id_to_title, top_n=30)

print(f"\n🎬 Top 30 Recommendations for mood '{mood}' for User Index {user_idx}:")
for i, title in enumerate(recommendations, 1):
    print(f"{i}. {title}")

real_user_id = train_set.idx2userid[user_idx]
print(f"\nUser index {user_idx} corresponds to user_id: {real_user_id}")
print("\nSurvey data for that user:")
print(survey_df[survey_df["user_id"] == real_user_id])
