import numpy as np
import pandas as pd
import warnings
import spotipy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict
warnings.filterwarnings("ignore")

# Read Spotify dataset
data = pd.read_csv("./spotify/data.csv")
data_moods = pd.read_csv("./spotify/data_moods.csv")
genre_data = pd.read_csv('./spotify/data_by_genres.csv')
data['decade'] = data['year'].apply(lambda year : f'{(year//10)*10}s' )

# Cluster the song genre data
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=12))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=25, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Connect to Spotify dashborad
def connect_spotify_dashboard():
    CLIENT_ID = ''   #spotify dashboard client id
    CLIENT_SECRET = ''  #spotify dashboard client secret
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, 
                                                               client_secret=CLIENT_SECRET))
    return sp

# Search song on Spotify
def find_song(name, year):
    sp = connect_spotify_dashboard()
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value
    return pd.DataFrame(song_data)

# Get song data for local dataset or Spotify dataset
def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        print('Fetching song information from local dataset')
        return song_data
    
    except IndexError:
        print('Fetching song information from spotify dataset')
        return find_song(song['name'], song['year'])

# Calculate mean vector
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0) 

# Flatten data to dictionary list
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys(): 
        flattened_dict[key] = [] 
    for dic in dict_list:
        for key,value in dic.items():
            flattened_dict[key].append(value) 
    return flattened_dict

# Recommend song with song name and song year
def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1] 
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

# login Spotify for create playlist
def login_spotify_Auth():
    scope = "playlist-modify-private"  
    username = "" #spotify user name
    client_id = ''  # spotify dashboard client id
    client_secret = ''  #spotify dashboard client secret
    redirect_uri = "http://localhost:8888/callback"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, username=username, client_id=client_id,
                                                client_secret=client_secret,redirect_uri=redirect_uri))
    return sp, username

# Get recommend playlist 
def get_recommend_playlist(recommend_song,year):
    global data
    recommend_song_list = recommend_songs([{'name': recommend_song, 'year':year}],  data)
    recommend_song_to_user = {'Name':[],'Year':[],'Artists':[]}
    for i in recommend_song_list:
        recommend_song_to_user['Name'].append(i['name'])
        recommend_song_to_user['Year'].append(i['year'])
        recommend_song_to_user['Artists'].append(i['artists'])
    sp, username = login_spotify_Auth()
    
    track_id = []

    for i in recommend_song_to_user["Name"]:
        track = sp.search(q=i, type="track", limit=1)
        track_id.append(track['tracks']['items'][0]['id'])

    playlist = sp.user_playlist_create(user=username, name="Recommend Play List", public=False)
    playlist_id = playlist['id']

    for i in track_id:
        sp.playlist_add_items(playlist_id=playlist['id'], items=[i])

    playlist = sp.user_playlist(username, playlist_id)
    playlist_link = playlist['external_urls']['spotify']
    print("Playlist link：", playlist_link)
    return playlist_link

# Get recommend song with mood
def get_recommend_moods(result):
    if result == 2:
        random = data_moods[data_moods['mood'] == 'Energetic']
        random_row = random.sample()
        date = random_row['release_date'].values[0]
        url = get_recommend_playlist(random_row['name'].values[0],int(date[0:4]))
    elif result == 1:
        random = data_moods[data_moods['mood'] == 'Happy']
        random_row = random.sample()
        date = random_row['release_date'].values[0]
        url = get_recommend_playlist(random_row['name'].values[0],int(date[0:4]))
    elif result == 0:
        random = data_moods[data_moods['mood'] == 'Calm']
        random_row = random.sample()
        date = random_row['release_date'].values[0]
        url = get_recommend_playlist(random_row['name'].values[0],int(date[0:4]))
    elif result == -1 or result == -2:
        random = data_moods[data_moods['mood'] == 'Sad']
        random_row = random.sample()
        date = random_row['release_date'].values[0]
        url = get_recommend_playlist(random_row['name'].values[0],int(date[0:4]))
    return url