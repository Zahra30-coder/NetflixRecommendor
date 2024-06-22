import pickle
import streamlit as st
import requests
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

des_embeddings = np.load("des_embeddings.pt")
df = pickle.load(open('df.pkl', 'rb'))

def recommend(query_title, df):
    # Get show description based on title
    query_title = query_title.strip()  # Remove leading/trailing spaces
    query_show_des = df[df['title'].str.strip() == query_title]['description'].to_list()[0]

    # Encode the query description and calculate cosine similarities
    query_embedd = model.encode(query_show_des)
    cosine_scores = util.pytorch_cos_sim(query_embedd, des_embeddings)

    # Find top 5 most similar shows (titles)
    top5_matches = torch.argsort(cosine_scores, dim=-1, descending=True).tolist()[0][1:6]
    similar_titles = df.iloc[top5_matches]['title'].to_list()

    # Create and return DataFrame
    result_df = pd.DataFrame({'title': similar_titles})

    # Save embeddings (des_embeddings) to a file
    torch.save(des_embeddings, 'des_embeddings.pt')

    return result_df

st.set_page_config(layout="wide")

st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 70%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
    
</style>
""",
    unsafe_allow_html=True,
)
#################

header_container = st.container()

from PIL import Image
# arxiv_img = Image.open('arxiv.png')

with header_container:
    # for example a logo or a image that looks like a website header

    # different levels of text you can include in your app
    st.title("Movies Recommendor")
    # st.header("Welcome!")
    #st.image(PIL.Image.open('arxiv.png'))
    # st.subheader("This is a great app")
    st.write(
        "Searching for the right Movie to watch ?",
        "Your search ends here !!! \U0001F642")

selected_keyword = st.text_input(
    "Type Movie Name", 'Stranger Things'
)

recommendded_results = recommend(selected_keyword)
# l=[]
if st.button('Show Recommendation'):

    for index in recommendded_results:
        # st.text(df.iloc[index,:])
        # st.text(df.iloc[index].iloc[0])
        # l.append(df.iloc[index].iloc[0])
        with st.expander(df.iloc[index].iloc[3]):
            st.write("Title :  ",df.iloc[index].iloc[2])
            st.write("Content Type :  ",df.iloc[index].iloc[2])

            st.write("Duration :  ",df.iloc[index].iloc[-3])

            st.write("Rating :  ",df.iloc[index].iloc[-4])
            st.write("Release Year :  ",df.iloc[index].iloc[-5])
            st.write("Desctiption :  ",df.iloc[index].iloc[-1])
            
            
            
    '''
    
    .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2022/10/11/21/58/texture-7515225__340.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
    
    
    '''
# run using python app.py
#streamlit run app.py