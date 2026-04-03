import os
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth, SpotifyPKCE
import pandas as pd
from dotenv import load_dotenv

# Config

load_dotenv()
CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID",     "YOUR_CLIENT_ID_HERE")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")

TARGET_SONGS  = 10000   # Aim for this many unique tracks
OUTPUT_FILE   = "songs.csv"
BATCH_SIZE    = 100      # Max batch size allowed by API

# Source playlists to check
PLAYLIST_IDS = [
    # Genres
    "37i9dQZF1DWXRqgorJj26U",  # Rock classics
    "37i9dQZF1DX0XUsuxWHRQd",  # Rap caviar
    "37i9dQZF1DX4dyzvuaRJ0n",  # Pop hits
    "37i9dQZF1DXbITWG1ZJKYt",  # Jazz classics
    "37i9dQZF1DWXRqgorJj26U",  # R&B classics
    "37i9dQZF1DWWEJlAGA9gs0",  # Classical essentials
    "37i9dQZF1DX76Wlfdnj7AP",  # Beast mode (metal/rock)
    "37i9dQZF1DX4UtSsGT1Sbe",  # All out 80s pop
    "37i9dQZF1DX1lVhptIYRda",  # Hot country
    "37i9dQZF1DX9tPFAsjuVZb",  # EDM/electronic
    "37i9dQZF1DX8tZsk68tuDw",  # Latin hits
    "37i9dQZF1DWZeKCadgRdKQ",  # Blues classics
    "37i9dQZF1DX4SBhb3fqCJd",  # African heat
    "37i9dQZF1DX3rxVfibe1L0",  # Mood booster
    # Decade playlists
    "37i9dQZF1DX4UtSsGT1Sbe",  # All out 80s
    "37i9dQZF1DXbG22YGu08Bs",  # All out 70s
    "37i9dQZF1DX4o1uurxF9Si",  # All out 60s
    "37i9dQZF1DX3oM43U2C7Jr",  # All out 90s
    "37i9dQZF1DX4dyzvuaRJ0n",  # All out 2000s
    # Mood playlists
    "37i9dQZF1DX3rxVfibe1L0",  # Mood booster
    "37i9dQZF1DWXLeA8Omikj7",  # Brain food (focus)
    "37i9dQZF1DX889U0CL85jj",  # Sleep
    "37i9dQZF1DX6VdMW310YC7",  # Chill hits
    "37i9dQZF1DX2UgsUIyXVjf",  # Workout banger
    "37i9dQZF1DWUa8ZRTfalHk",  # Party hits
    "37i9dQZF1DX4WYpdgoIcn6",  # Chill vibes
    # Chart/viral playlists
    "37i9dQZEVXbMDoHDwVN2tF",  # Global top 50
    "37i9dQZEVXbLiRSasKsNU9",  # US viral 50
    "37i9dQZF1DX0kbJZpiYdZl",  # Hot hits UK
]

# Audio features to keep
FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
]

# Helper functions

def get_playlist_track_ids(sp, playlist_id):
    """Return all track IDs from a playlist."""
    ids = []
    try:
        results = sp.playlist_items(
            playlist_id,
            fields="items.track.id,next",
            additional_types=["track"],
            limit=100,
            market = "IE",
        )
        while results:
            for item in results["items"]:
                track = item.get("track")
                if track and track.get("id"):
                    ids.append(track["id"])
            results = sp.next(results) if results.get("next") else None
    except Exception as e:
        print(f"  Warning: could not fetch playlist {playlist_id}: {e}")
    return ids


def get_audio_features(sp, track_ids):
    """
    Fetch audio features for a list of track IDs in batches of BATCH_SIZE.
    Returns a list of feature dicts (None entries are skipped).
    """
    features = []
    for i in range(0, len(track_ids), BATCH_SIZE):
        batch = track_ids[i : i + BATCH_SIZE]
        try:
            results = sp.audio_features(batch)
            features.extend([r for r in results if r is not None])
        except Exception as e:
            print(f"  Warning: audio features batch failed: {e}")
        time.sleep(0.1)  # Stay well inside rate limits
    return features


def get_track_metadata(sp, track_ids):
    """
    Fetch track name, artist, album, and popularity for a list of IDs.
    Returns a dict keyed by track ID.
    """
    metadata = {}
    for i in range(0, len(track_ids), 50):   # tracks endpoint max = 50
        batch = track_ids[i : i + 50]
        try:
            results = sp.tracks(batch)
            for track in results["tracks"]:
                if track:
                    metadata[track["id"]] = {
                        "track_name":  track["name"],
                        "artist_name": track["artists"][0]["name"] if track["artists"] else "",
                        "artist_id":   track["artists"][0]["id"]   if track["artists"] else None,
                        "album_name":  track["album"]["name"],
                        "release_date": track["album"].get("release_date", ""),
                        "popularity":  track["popularity"],
                    }
        except Exception as e:
            print(f"  Warning: metadata batch failed: {e}")
        time.sleep(0.1)
    return metadata

def get_artist_features(sp, artist_ids):
    """
    Fetch follower count and genres for a list of artist IDs.
    Returns a dict keyed by artist ID.
    """
    artist_data = {}
    unique_ids = list(set(artist_ids))          # artists repeat across tracks
    for i in range(0, len(unique_ids), 50):     # artists endpoint max = 50
        batch = unique_ids[i : i + 50]
        try:
            results = sp.artists(batch)
            for artist in results["artists"]:
                if artist:
                    artist_data[artist["id"]] = {
                        "artist_followers": artist["followers"]["total"],
                        "artist_popularity": artist["popularity"],  # Spotify's own 0-100 score
                        "artist_genres": "|".join(artist["genres"]),  # pipe-separated, e.g. "pop|dance pop"
                    }
        except Exception as e:
            print(f"  Warning: artist batch failed: {e}")
        time.sleep(0.1)
    return artist_data

# Main scraping loop

def main():
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri="http://127.0.0.1:8888/callback",
        scope=None,
        open_browser=False,
    )

    # Print the URL and ask the user to paste back the redirected URL
    auth_url = auth_manager.get_authorize_url()
    print(f"\nOpen this URL in your browser:\n{auth_url}\n")
    print("After authorising, your browser will redirect to a URL starting with")
    print("http://127.0.0.1:8888/callback?code=...")
    print("Paste that full URL here (the page itself can show an error, that's fine):\n")

    redirected_url = input("Paste URL: ").strip()
    code = auth_manager.parse_response_code(redirected_url)
    auth_manager.get_access_token(code, as_dict=False)

    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Step 1: collect track IDs from all playlists
    print(f"\nCollecting track IDs from {len(PLAYLIST_IDS)} playlists...")
    all_ids = set()
    for pid in PLAYLIST_IDS:
        ids = get_playlist_track_ids(sp, pid)
        before = len(all_ids)
        all_ids.update(ids)
        print(f"  Playlist {pid[:22]}… → {len(ids)} tracks "
              f"({len(all_ids) - before} new, {len(all_ids)} total unique)")

    all_ids = list(all_ids)
    print(f"\nTotal unique track IDs: {len(all_ids)}")

    # Trim if we have more than needed (keeps runtime reasonable)
    if len(all_ids) > TARGET_SONGS:
        import random
        random.shuffle(all_ids)
        all_ids = all_ids[:TARGET_SONGS]
        print(f"Trimmed to {TARGET_SONGS} tracks for target dataset size.")

    # Step 2: fetch audio features
    print(f"\nFetching audio features (batches of {BATCH_SIZE})...")
    features = get_audio_features(sp, all_ids)
    print(f"  Got features for {len(features)} tracks.")

    # Step 3: fetch metadata (name, artist, popularity)
    feature_ids = [f["id"] for f in features]
    print(f"\nFetching track metadata (popularity, artist, etc.)...")
    metadata = get_track_metadata(sp, feature_ids)
    print(f"  Got metadata for {len(metadata)} tracks.")
    # Step 3b: fetch artist features (followers, popularity, genre)
    print(f"\nFetching artist features...")
    artist_ids = [metadata[tid]["artist_id"] for tid in feature_ids if tid in metadata]
    artist_features = get_artist_features(sp, artist_ids)
    print(f"  Got artist data for {len(artist_features)} unique artists.")

    # Step 4: merge features + metadata into a DataFrame
    rows = []
    for f in features:
        tid = f["id"]
        if tid not in metadata:
            continue
        row = {"track_id": tid}
        row.update(metadata[tid])
        aid = metadata[tid].get("artist_id")
        row.update(artist_features.get(aid, {}))   # gracefully empty if artist lookup failed
        row.update({col: f.get(col) for col in FEATURE_COLS})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Step 5: clean up
    before = len(df)
    df.drop_duplicates(subset="track_id", inplace=True)
    df.dropna(subset=FEATURE_COLS + ["popularity"], inplace=True)
    print(f"\nDropped {before - len(df)} duplicate/null rows. "
          f"Final dataset: {len(df)} songs.")

    # Step 6: save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"\nPopularity distribution:\n{df['popularity'].describe().round(2)}")
    print(f"\nFeature summary:\n{df[FEATURE_COLS].describe().round(3)}")


if __name__ == "__main__":
    main()