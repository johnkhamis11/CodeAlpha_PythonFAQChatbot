import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = [
    {"question": "How do I install Python?", "answer": "You can install Python from the official website python.org/downloads. Choose the right installer for your OS."},
    {"question": "What is a variable in Python?", "answer": "A variable stores data values. You can create one by assigning a value: x = 5."},
    {"question": "How do I write a comment in Python?", "answer": "Use the # symbol. For example: # This is a comment."},
    {"question": "What is a list in Python?", "answer": "A list is a collection of items. You can create one with square brackets: my_list = [1, 2, 3]."},
    {"question": "How do I print something in Python?", "answer": "Use the print() function. Example: print('Hello, World!')."},
    {"question": "How do I install packages in Python?", "answer": "Use pip, the Python package installer. Example: pip install package_name."},
    {"question": "How do I define a function in Python?", "answer": "Use the def keyword. Example: def my_function(): pass."},
    {"question": "What is a dictionary in Python?", "answer": "A dictionary stores key-value pairs. Example: my_dict = {'key': 'value'}."},
]

st.title("Python FAQs Chatbot 🤖")

user_question = st.text_input("Ask me a question about Python basics:")

if st.button("Get Answer"):
    questions = [faq["question"] for faq in faqs]
    vectorizer = TfidfVectorizer().fit_transform(questions + [user_question])
    vectors = vectorizer.toarray()

    user_vec = vectors[-1]
    faq_vecs = vectors[:-1]

    similarities = cosine_similarity([user_vec], faq_vecs).flatten()
    best_match_index = np.argmax(similarities)
    best_faq = faqs[best_match_index]

    st.write(f"**Best Answer:** {best_faq['answer']}")
