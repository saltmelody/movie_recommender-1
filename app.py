import streamlit as st
import pandas as pd
import numpy as np
import gensim

st.title('映画レコメンド')

# 映画評価データの読み込み
movielens = pd.read_csv("movielens.tsv", sep="\t")

# 学習済みのitem2vecモデルの読み込み
model = gensim.models.word2vec.Word2Vec.load("item2vec.model")

# 映画評価データの確認
st.write("映画評価データ(確認用)")
st.write(movielens.head(5))

# 映画IDとタイトルを辞書型に変換
movies = movielens["title"].unique()
movie_ids = movielens["movie_id"].unique()
movie_id_to_title = dict(zip(movie_ids, movies))
movie_title_to_id = dict(zip(movies, movie_ids))

st.markdown("## 1本の映画に対して似ている映画を表示する")
selected_movie = st.selectbox("映画を選んでください", movies)
selected_movie_id = movie_title_to_id[selected_movie]
st.write(f"あなたが選択した映画は{selected_movie}(id={selected_movie_id})です")

# 似ている映画を表示
st.markdown(f"### {selected_movie}に似ている映画")
results = []
for movie_id, score in model.wv.most_similar(selected_movie_id):
    title = movie_id_to_title[movie_id]
    results.append({"movie_id":movie_id, "title": title, "score": score})
results = pd.DataFrame(results)
st.write(results)


st.markdown("## 複数の映画を選んでおすすめの映画を表示する")

selected_movies = st.multiselect("映画を複数選んでください", movies)
selected_movie_ids = [movie_title_to_id[movie] for movie in selected_movies]
vectors = [model.wv.get_vector(movie_id) for movie_id in selected_movie_ids]
if len(selected_movies) > 0:
    user_vector = np.mean(vectors, axis=0)
    st.markdown(f"### おすすめの映画")
    recommend_results = []
    for movie_id, score in model.wv.most_similar(user_vector):
        title = movie_id_to_title[movie_id]
        recommend_results.append({"movie_id":movie_id, "title": title, "score": score})
    recommend_results = pd.DataFrame(recommend_results)
    st.write(recommend_results)
