"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["HOME","ABOUT US","Recommender System","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("We the FSC_TECH company data science company offers solutions that leverage data analytics, machine learning,")
        st.write(" and artificial intelligence to solve complex problems, also recommend and provide valuable insights to clients. ")
        st.write("Here's a detailed overview of the various components and processes involved in the operations of FSC_TECH company:")
        st.write("A: Problem Identification and Consultation")
        st.write("B: Data Acquisition and Integration")
        st.write("C: Exploratory Data Analysis (EDA)")
        st.write("D: Feature Engineering")
        st.write("E: Model Development")
        st.write("F: Model Evaluation and Validation")
        st.write("G: Deployment and Integration")
        st.write("H: Continuous Monitoring and Maintenance")
        st.write("I: Reporting and Visualization")
        st.write("J: Ongoing Support and Collaboration")

    if page_selection == "HOME":
        st.title("HOME")
        st.write("FSC_TECH company are into building recommendation system for companies and also offers the best services for our clients.")

    if page_selection == "ABOUT US":
        st.title("ABOUT US")
        st.write("FSC_TECH company is a business that specializes in utilizing advanced analytics techniques, machine learning,")
        st.write("and statistical modeling to extract valuable insights from data. These insights can be used to make data-driven decisions,")
        st.write("solve complex problems, and drive business growth.")
        st.write("Here are some key aspects and details about FSC_TECH company:")
        st.write("A: Services: we offer a range of services to clients, including,Data Analysis,Predictive Analytics,Data Visualization,")
        st.write("Machine Learning Development, Data Engineering and Consulting and Strategy.")
        st.write("B: Expertise: we typically have a team of highly skilled professionals with expertise in various domains, including:")
        st.write(" Data Scientists,Machine Learning Engineers, Data Engineers, Domain Experts,Data Visualization Specialists.")
        st.write("C: Tools and Technologies: We utilize a wide range of tools and technologies to perform our work effectively.")
        st.write("Some common ones include: Programming Languages,Machine Learning Frameworks,Big Data Technologies,Data Visualization Tools")
        st.write("Cloud Platforms.")
        st.write("Client Engagement: A FSC_TECH company typically engages with clients through a systematic process that involves:")
        st.write("Requirement Gathering, Data Exploration,Modeling and Analysis,Visualization and Reporting,Iteration and Improvement.")
        st.write("D: Data Privacy and Security: FSC_TECH company prioritize data privacy and security,we ensure that client data is handled")
        st.write("securely, adhere to relevant data protection regulations, and implement best practices for data anonymization and encryption")
        st.write("E: Industry Applications: we work across various industries, includingfinance, healthcare, e-commerce, manufacturing,")
        st.write("transportation, and marketing.we help businesses in these sectors optimize operations, improve customer targeting, detect fraud")
        st.write("enhance forecasting accuracy, and support decision-making processes.")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
