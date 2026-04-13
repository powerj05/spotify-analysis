import pandas as pd

# Config
TRACKS_FILE  = "featured_Spotify_track_info.csv"
ARTISTS_FILE = "featured_Spotify_artist_info.csv"
OUTPUT_FILE  = "songs_merged.csv"
SAMPLE_SIZE  = 10_000
RANDOM_SEED  = 42

TRACK_ARTIST_ID_COL  = "artists"   # artist ID column in the tracks file
ARTIST_ID_COL        = "ids"   # artist ID column in the artists file

print("Loading data...")
tracks  = pd.read_csv(TRACKS_FILE)
artists = pd.read_csv(ARTISTS_FILE)
print(f"  Tracks:  {len(tracks)} rows, columns: {list(tracks.columns)}")
print(f"  Artists: {len(artists)} rows, columns: {list(artists.columns)}")

merged = tracks.merge(artists, left_on=TRACK_ARTIST_ID_COL, right_on=ARTIST_ID_COL, suffixes=("", "_artist"))
print(f"\nAfter merge: {len(merged)} rows.")

# Sample 10k songs

n = min(SAMPLE_SIZE, len(merged))
sample = merged.sample(n=n, random_state=RANDOM_SEED)
print(f"Sampled {len(sample)} rows.")

sample.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
print(f"Final columns: {list(sample.columns)}")