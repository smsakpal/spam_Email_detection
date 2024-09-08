import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)

    Y = []
    for i in Message:
        if i.isalnum():
            Y.append(i)

    Message = Y[:]
    Y.clear()

    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            Y.append(i)

    Message = Y[:]
    Y.clear()

    for i in Message:
        Y.append(ps.stem(i))

    return " ".join(Y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
svm_model = pickle.load(open('model.pkl','rb'))


st.title("Spam Email Identifier")

input_email = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_email = transform_text(input_email)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_email])
    vector_input_dense = vector_input.toarray()
    # 3. predict
    result = svm_model.predict(vector_input_dense)[0]
    # 4. Display
    if result == 0:
        st.header("Spam Email")
    else:
        st.header("Ham Email")