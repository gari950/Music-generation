## driver file

import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

st.set_page_config(page_title="Music Generation App",layout="wide")
#st.markdown('<style> [theme] backgroundColor="#FFFFFF" </style>', unsafe_allow_html=True)
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Enter a prompt")
text = st.text_area("for example: a chill song with influences from lofi, chillstep and downtempo")
button = st.button("Submit", key="button")
if button:
    audio_bytes = query({
        "inputs": text,
    })
    # You can access the audio with IPython.display for example
    from IPython.display import Audio
    st.title("Generated music sample:")
    out = Audio(audio_bytes)
    st.audio(audio_bytes)