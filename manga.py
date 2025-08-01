import streamlit as st
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
import numpy as np

# ── 読み込み ───────────────────────────
vectors = np.load("data/my_vectors.npy")                    # shape = (V, D)
vocab   = np.load("data/my_vocab.npy", allow_pickle=True)   # shape = (V,)

# ── KeyedVectors を組み立てる ─────────
kv = KeyedVectors(vector_size=vectors.shape[1])
kv.add_vectors(vocab.tolist(), vectors)      # gensim 4.x 以降の公式 API :contentReference[oaicite:0]{index=0}
kv.fill_norms()                              # 類似度計算を高速化（省略可）


st.write("マンガレコメンドアプリ")

manga_titles = vocab.tolist()

st.markdown("## 1冊のマンガに対して似ているマンガを表示する")
selected_manga = st.selectbox("マンガを選んでください", manga_titles)

# 似ている映画を表示
st.markdown(f"### {selected_manga}に似ているマンガ")
results = []
for recommend_manga, score in kv.most_similar(selected_manga,topn=30):
    results.append({"title": recommend_manga, "score": score})
results = pd.DataFrame(results)
st.write(results)


# 本来なら下記のような簡単な読み込みで対抗可能。ただし、gensimバージョンが異なるとエラー出る
# model = gensim.models.word2vec.Word2Vec.load("data/manga_item2vec.model")

#複数の漫画
st.markdown("## 複数の漫画を選んでおすすめの漫画を表示する")

selected_mangas = st.multiselect("漫画を複数選んでください", manga_titles)

if selected_mangas:
    vectors = [kv.get_vector(manga) for manga in selected_mangas]
    user_vector = np.mean(vectors, axis=0)
    st.markdown(f"### おすすめのマンガ")
    recommend_results = []
    for manga, score in kv.similar_by_vector(user_vector, topn=30):
        recommend_results.append({"title": manga, "score": score})
    recommend_results = pd.DataFrame(recommend_results)
    st.write(recommend_results)