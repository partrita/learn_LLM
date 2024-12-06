# Tokens and embeddings

# let's look at how we can generate contextualized word embeddings

from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# load a languuage model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# tokenize the sentence
tokens = tokenizer("Hello world", return_tensors="pt")

#process the tokens
output = model(**tokens)[0]

# Training a song embedding model
import pandas as pd
from urllib import request

# get the playlist dataset file
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

# Parse the playlist dataset file. Skip the first two lines as
# they only contain metadata
lines = data.read().decode("utf-8").split('\n')[2:]

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

# let's train the model
from gensim.models import Word2Vec
# train our word2vec model
model = Work2Vec(
        playlists, vector_size=32, window=20,
        negative=50, min_count=1, workers=4
)

# ask the model for songs
def print_recommendations(song_id):
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id),topn=5)
    )[:,0]
    return  songs_df.iloc[similar_songs]

# Extract recommendations
print_recommendations(2172)
