import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import pickle

# Setting up the app configuration
st.set_page_config(
    page_title="AI-Based Disease & Treatment Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Header
st.title("AI-Based Disease & Treatment Advisor")
st.subheader("Predict, Prevent, and Personalize Your Health Journey.")

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precaution = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medication = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open("models/svc.pkl", "rb"))

# The helper function
def helper(dis):
    # Standardize the disease name
    dis = dis.strip().lower()

    # Extract description
    desc = description[description["Disease"].str.strip().str.lower() == dis]["Description"]
    desc = "".join(desc.values) if not desc.empty else "Description not available."

    # Extract precautions
    prec = precaution[precaution["Disease"].str.strip().str.lower() == dis][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
    prec = prec.values.tolist() if not prec.empty else [["No precautions available."]]

    # Extract medications
    medic = medication[medication["Disease"].str.strip().str.lower() == dis]["Medication"]
    medic = medic.values.tolist() if not medic.empty else ["No medications available."]

    # Extract diets
    diet = diets[diets["Disease"].str.strip().str.lower() == dis]["Diet"]
    diet = diet.values.tolist() if not diet.empty else ["No diet recommendations available."]

    # Extract workouts
    work = workout[workout["disease"].str.strip().str.lower() == dis]["workout"]
    work = work.values.tolist() if not work.empty else ["No workout recommendations available."]

    return desc, prec, medic, diet, work

# Symptoms and diseases dictionaries
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 
                 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 
                 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 
                 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 
                 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 
                 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 
                 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 
                 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 
                 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 
                 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 
                 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 
                 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 
                 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 
                 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 
                 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 
                 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 
                 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 
                 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 
                 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 
                 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 
                 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 
                 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 
                 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}



# Prediction with proper feature names
def get_predicted_value(selected_symptoms):
    # Initialize an array for the symptoms
    symptoms_input = np.zeros(len(symptoms_dict))

    # Mark the selected symptoms as 1
    for symptom in selected_symptoms:
        symptoms_input[symptoms_dict[symptom]] = 1

    # Convert symptoms_input to a DataFrame
    feature_names = list(symptoms_dict.keys())
    symptoms_input_df = pd.DataFrame([symptoms_input], columns=feature_names)

    # Use the model to predict
    prediction = svc.predict(symptoms_input_df)
    predicted_disease = diseases_list.get(prediction[0], "Unknown Disease")

    return predicted_disease


# Side bar navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "Contact", "Developer"])
    return page


# Home page content
def home_page():
    # Symptom input using multiselect
    selected_symptoms = st.multiselect(
        "Enter the patient's symptoms below to predict the disease.",
        options=list(symptoms_dict.keys()),
        help="Choose one or more symptoms from the list",
    )

    # Create button layout
    button_style = """
    <style>
        .stButton>button {
            width: 120px; 
            height: 40px; 
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Button placeholders
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    with col1:
        disease_button = st.button("Disease")
    with col2:
        desc_button = st.button("Description")
    with col3:
        precaution_button = st.button("Precautions")
    with col4:
        medication_button = st.button("Medications")
    with col5:
        diet_button = st.button("Diet")
    with col6:
        workout_button = st.button("Workout")

    # Handle no symptoms selected
    if not selected_symptoms:
        st.info("Please select symptoms to see predictions and recommendations.")
    else:
        # Get prediction
        predicted_disease = get_predicted_value(selected_symptoms)
        desc, prec, medic, diet, work = helper(predicted_disease)

        # Display results
        if disease_button:
            with st.expander("**Predicted Disease:**", expanded=True):
                st.write(f"{predicted_disease}")

        if desc_button:
            with st.expander("**Description:**", expanded=True):
                st.write(desc)

        if precaution_button:
            with st.expander("**Precautions:**", expanded=True):
                prec_clean = [p for p in prec[0] if str(p) != 'nan']
                for i, precaution in enumerate(prec_clean, 1):
                    st.write(f"{i}. {precaution}")

        if medication_button:
            with st.expander("**Medications:**", expanded=True):
                medic_text = ", ".join(medic)
                st.write(f"{medic_text}")

        if diet_button:
            with st.expander("**Diet Recommendations:**", expanded=True):
                diet_text = ", ".join(diet)
                st.write(f"{diet_text}")

        if workout_button:
            with st.expander("**Workout Recommendations:**", expanded=True):
                for i, workout in enumerate(work, 1):
                    st.write(f"{i}. {workout}")


# About page content
def about_page():
    st.title("About Application")
    st.write("""

    Welcome to the Disease Prediction App! This intelligent tool leverages machine learning to predict potential diseases based on the symptoms you provide. 
    Designed to assist users in understanding their health better, the app offers a comprehensive suite of features, including detailed descriptions of the predicted disease, 
    suggested precautions, recommended medications, tailored diet plans, and suitable workout routines. Whether you're looking for early insights into your symptoms or 
    just want to stay informed, this app empowers you to make proactive and informed health decisions.

    ### How to use:
    1. Enter the patient's symptoms in the input field.
    2. The app will predict the most likely disease based on the entered symptoms.
    3. It will then provide additional information about the disease, including description, precautions, medications, and more.
    """)

# Contact page content
def contact_page():
    st.title("Contact Us")
    # Personal Description Section
    st.write("""
        Hello! I'm Muhammad Hamza, a passionate individual dedicated to improving various industries through technology and data science. 
        With a background in data science & AI, I'm focused on utilizing AI and machine learning to enhance services and provide innovative solutions across multiple domains. 
        If you have any questions or would like to connect, feel free to reach out to me.
    """)

    # Contact Information
    st.write("**Email:** mr.hamxa942@gmail.com")
    st.write("**Phone:** +92 335 9588458")

    # Location
    st.subheader("Location")
    st.write("Karak, KPK, Pakistan")
    
# Developer page content
def developer_page():
    st.header("👨‍💻 Data Science & AI Developer")
    st.write("Developed by Muhammad Hamza.")
    st.markdown("""
    **Connect with me**:
    - [GitHub](https://github.com/mrhamxo)
    - [LinkedIn](linkedin.com/in/muhammad-hamza-khattak/)
    """)
    st.write("""
    I'm a passionate developer with expertise in Data Science and AI, 
    always eager to create impactful solutions.
    """)
    
# Footer content
def footer():
    st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer {
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        <p>Created with ❤️ by Muhammad Hamza</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Add image to the top left sidebar
    st.sidebar.image("static//img.png", width=150)

    page = sidebar_navigation()
    
    # Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**© 2024. All Rights Reserved.**")
    
    if page == "Home":
        home_page()
    elif page == "About":
        about_page()
    elif page == "Contact":
        contact_page()
    elif page == "Developer":
        developer_page()

    footer()

if __name__ == "__main__":
    main()
