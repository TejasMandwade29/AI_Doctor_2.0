# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

#load_dotenv()

# PRE-DEFINED SYMPTOM SOLUTIONS DATABASE (ZERO API COST)
SYMPTOM_SOLUTIONS = {
    # Fever & Cold Patterns
    "fever_cold": {
        "symptoms": ["Fever", "Cough/Cold", "Chills", "Fatigue"],
        "condition": "Common Cold or Viral Infection",
        "advice": "Rest, drink plenty of fluids, and take over-the-counter fever reducers like acetaminophen. Monitor your temperature and see a doctor if fever persists beyond 3 days.",
        "urgency": "Moderate - See doctor if no improvement in 3 days"
    },
    
    "flu": {
        "symptoms": ["Fever", "Muscle Pain", "Chills", "Fatigue", "Headache"],
        "condition": "Influenza (Flu)",
        "advice": "Get plenty of rest, stay hydrated, and consider antiviral medication if diagnosed early. Isolate to prevent spreading.",
        "urgency": "Moderate - Consult doctor within 24-48 hours"
    },
    
    # Pain Patterns
    "migraine": {
        "symptoms": ["Headache", "Vision Problems", "Nausea"],
        "condition": "Migraine Headache",
        "advice": "Rest in a dark, quiet room. Apply cold compress to forehead. Stay hydrated and avoid triggers like bright lights or strong smells.",
        "urgency": "Moderate - See doctor for recurring migraines"
    },
    
    "arthritis": {
        "symptoms": ["Joint Pain", "Muscle Pain"],
        "condition": "Arthritis or Joint Inflammation",
        "advice": "Apply ice packs to affected joints, avoid strenuous activities. Consider over-the-counter anti-inflammatory medication. Gentle stretching may help.",
        "urgency": "Low - Schedule doctor appointment"
    },
    
    "muscle_strain": {
        "symptoms": ["Muscle Pain", "Fatigue"],
        "condition": "Muscle Strain or Overuse",
        "advice": "Rest the affected area, apply ice for 20 minutes several times daily. Gentle stretching after 48 hours. Avoid activities that cause pain.",
        "urgency": "Low - Self-care for 3-5 days"
    },
    
    # Respiratory Patterns
    "allergies": {
        "symptoms": ["Cough/Cold", "Fatigue"],
        "condition": "Seasonal Allergies",
        "advice": "Avoid allergens, use over-the-counter antihistamines. Keep windows closed during high pollen days. Consider air purifier.",
        "urgency": "Low - Try OTC allergy medication first"
    },
    
    # General Illness
    "fatigue_syndrome": {
        "symptoms": ["Fatigue", "Sleep Issues", "Appetite Loss"],
        "condition": "Fatigue or Stress-Related Condition",
        "advice": "Ensure 7-8 hours of sleep nightly, maintain regular sleep schedule. Eat balanced meals and stay hydrated. Reduce stress through relaxation techniques.",
        "urgency": "Low - Lifestyle changes recommended"
    },
    
    "dehydration": {
        "symptoms": ["Fatigue", "Dizziness", "Headache"],
        "condition": "Possible Dehydration",
        "advice": "Drink water consistently throughout the day. Include electrolyte solutions if sweating heavily. Avoid caffeine and alcohol.",
        "urgency": "Low - Increase fluid intake immediately"
    },
    
    "stomach_issues": {
        "symptoms": ["Nausea", "Appetite Loss", "Fatigue"],
        "condition": "Stomach Bug or Indigestion",
        "advice": "Stick to bland foods (BRAT diet: bananas, rice, applesauce, toast). Stay hydrated with small sips of water. Avoid spicy or fatty foods.",
        "urgency": "Low - See doctor if symptoms persist 48+ hours"
    }
}

def detect_condition_from_symptoms(selected_symptoms):
    """Match symptoms against pre-defined conditions without API calls"""
    if not selected_symptoms:
        return None
    
    # Clean symptom names (remove emojis)
    clean_symptoms = []
    for symptom in selected_symptoms:
        if ' ' in symptom:
            clean_symptoms.append(symptom.split(' ', 1)[1])
        else:
            clean_symptoms.append(symptom)
    
    symptom_set = set(clean_symptoms)
    
    best_match = None
    highest_match_count = 0
    
    # Find the best matching condition
    for condition_id, condition_data in SYMPTOM_SOLUTIONS.items():
        condition_symptoms = set(condition_data["symptoms"])
        matching_symptoms = symptom_set.intersection(condition_symptoms)
        match_count = len(matching_symptoms)
        
        # Require at least 2 matching symptoms for a valid match
        if match_count >= 2 and match_count > highest_match_count:
            highest_match_count = match_count
            best_match = condition_data
    
    return best_match

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath, selected_symptoms=None):
    # Step 1: Try pre-defined solutions first (NO API CALLS)
    predefined_solution = None
    if selected_symptoms:
        predefined_solution = detect_condition_from_symptoms(selected_symptoms)
    
    # Build symptom text
    symptom_text = ""
    if selected_symptoms:
        clean_symptoms = []
        for symptom in selected_symptoms:
            if ' ' in symptom:
                clean_symptoms.append(symptom.split(' ', 1)[1])
            else:
                clean_symptoms.append(symptom)
        symptom_text = "Patient reports symptoms: " + ", ".join(clean_symptoms) + ". "
        print(f"✅ Using selected symptoms: {symptom_text}")
    
    speech_to_text_output = ""
    
    # Handle audio input
    if audio_filepath:
        try:
            transcribed_text = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )
            speech_to_text_output = symptom_text + transcribed_text
        except Exception as e:
            speech_to_text_output = symptom_text + f"Error transcribing audio: {str(e)}"
    else:
        speech_to_text_output = symptom_text if symptom_text else "No symptoms described"
    
    # Step 2: Use pre-defined solution if available (SAVES API CALLS)
    if predefined_solution and not image_filepath:
        doctor_response = f"""🚨 QUICK DETECTION: {predefined_solution['condition']}

📋 SYMPTOMS MATCHED: {', '.join(predefined_solution['symptoms'])}

💡 RECOMMENDATIONS: {predefined_solution['advice']}

⚠️ URGENCY: {predefined_solution['urgency']}

Note: This is automated advice. Consult healthcare provider for proper diagnosis."""
        
        print("✅ Used pre-defined solution - Zero API cost!")
    
    # Step 3: Use AI for complex cases or with images
    elif image_filepath:
        try:
            doctor_response = analyze_image_with_query(
                query=system_prompt + (" " + speech_to_text_output if speech_to_text_output else ""), 
                encoded_image=encode_image(image_filepath), 
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        except Exception as e:
            doctor_response = f"Error analyzing image: {str(e)}"
    
    else:
        # Fallback to AI for symptoms without pre-defined match
        try:
            symptom_prompt = """You are a medical doctor for educational purposes. The patient reports: {symptoms}
            
            Provide brief possible causes and self-care advice in 2-3 sentences. Be concise and practical."""
            
            doctor_response = analyze_image_with_query(
                query=symptom_prompt.format(symptoms=speech_to_text_output), 
                encoded_image=None,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        except Exception as e:
            doctor_response = f"Based on your symptoms: {speech_to_text_output}. I recommend consulting a healthcare provider for proper diagnosis."

    # Generate voice response
    try:
        voice_of_doctor = text_to_speech_with_gtts(
            input_text=doctor_response, 
            output_filepath="final.mp3"
        )
    except Exception as e:
        voice_of_doctor = None
        print(f"Voice generation error: {e}")

    return speech_to_text_output, doctor_response, voice_of_doctor


# Custom CSS for better styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
}

.medical-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
    color: white !important;
    padding: 30px !important;
    border-radius: 15px !important;
    margin-bottom: 20px !important;
    text-align: center !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
}

.medical-card {
    background: white !important;
    padding: 25px !important;
    border-radius: 15px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08) !important;
    border-left: 5px solid #2a5298 !important;
    margin-bottom: 20px !important;
}

.warning-banner {
    background: linear-gradient(135deg, #fff3e0, #ffccbc) !important;
    border: 2px solid #ff5722 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    margin: 15px 0 !important;
}

.emergency-section {
    background: linear-gradient(135deg, #ffebee, #ffcdd2) !important;
    border: 2px solid #f44336 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    margin: 15px 0 !important;
}

.consult-btn {
    background: linear-gradient(135deg, #1e3c72, #2a5298) !important;
    border: none !important;
    color: white !important;
    padding: 15px 30px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}

.consult-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(42, 82, 152, 0.3) !important;
}

.input-section {
    background: white !important;
    border-radius: 12px !important;
    padding: 20px !important;
    border: 2px solid #e3f2fd !important;
    height: 100% !important;
}

.symptom-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 10px !important;
    margin: 15px 0 !important;
}

@media (max-width: 768px) {
    .symptom-grid {
        grid-template-columns: 1fr !important;
    }
}
"""

# Create the enhanced interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    ),
    title="AI Doctor 2.0 - Medical Analysis Assistant",
    css=custom_css
) as demo:
    
    # Header Section
    with gr.Column(elem_classes="medical-header"):
        gr.Markdown("""
        # 🩺 AI DOCTOR 2.0 
        ### Advanced Medical Analysis with Vision & Voice Technology
        *Powered by AI • For Educational Purposes Only*
        """)
    
    # Emergency Warning Section
    with gr.Column(elem_classes="emergency-section"):
        gr.Markdown("""
        # 🚨 EMERGENCY WARNING
        **Seek IMMEDIATE medical attention for:**
        - Chest pain or pressure • Difficulty breathing • Severe bleeding
        - Sudden weakness or numbness • Severe head injury • Suicidal thoughts
        
        **📞 Call emergency services immediately! This AI system cannot handle emergencies.**
        """)
    
    # Warning Notice
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            <div class="warning-banner">
            <h4 style='color: #c62828; margin-bottom: 10px;'>⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
            <p style='color: #b71c1c; margin: 0;'>This AI system is designed for educational and learning purposes only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult qualified healthcare providers for medical concerns.</p>
            </div>
            """)
    
    # Main Content Area
    with gr.Row(equal_height=True):
        # Input Section - Left Side
        with gr.Column(scale=1):
            with gr.Column(elem_classes="medical-card"):
                gr.Markdown("## 📥 PATIENT CONSULTATION")
                
                # QUICK SYMPTOM SELECTOR
                gr.Markdown("### 🔍 Quick Symptom Selector")
                gr.Markdown("*Select your symptoms to help with diagnosis*")
                
                common_symptoms = gr.CheckboxGroup(
                    choices=[
                        "🤒 Fever", "🤕 Headache", "🤧 Cough/Cold", "😵 Dizziness",
                        "💪 Muscle Pain", "🦵 Joint Pain", "🌡️ Chills", "🤢 Nausea",
                        "🥴 Fatigue", "🍽️ Appetite Loss", "😴 Sleep Issues",
                        "👁️ Vision Problems", "👂 Ear Pain", "🦷 Tooth Pain",
                        "📉 Weight Loss", "🧠 Memory Issues", "💓 Heart Palpitations"
                    ],
                    label="Select Your Symptoms",
                    info="Choose all symptoms that apply to you",
                    elem_classes="symptom-grid"
                )
    
    # Audio and Image Inputs Side by Side
    with gr.Row(equal_height=True):
        # Left Column - Audio Recording
        with gr.Column(scale=1):
            with gr.Column(elem_classes="input-section"):
                gr.Markdown("### 🎤 Record Additional Details")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record your symptoms or additional details",
                    show_download_button=True,
                    waveform_options={"show_controls": True}
                )
                gr.Markdown("""
                **💡 Recording Tips:**
                - Speak clearly and describe your symptoms
                - Mention when symptoms started
                - Keep recordings under 2 minutes
                """)
        
        # Right Column - Image Upload
        with gr.Column(scale=1):
            with gr.Column(elem_classes="input-section"):
                gr.Markdown("### 📷 Medical Imaging")
                image_input = gr.Image(
                    type="filepath",
                    label="Upload Medical Image",
                    height=300
                )
                
                # Example Images Section
                gr.Markdown("""
                **🖼️ Supported Image Types:**
                - Skin conditions & rashes
                - X-rays & MRI scans  
                - Injury photos
                - Eye problems
                - Mouth sores
                """)
    
    # Buttons Row
    with gr.Row():
        with gr.Column():
            with gr.Row():
                clear_btn = gr.Button("🗑️ CLEAR ALL", variant="secondary", size="lg", min_width=150)
                submit_btn = gr.Button("🔍 ANALYZE WITH AI DOCTOR", 
                                     variant="primary", 
                                     size="lg",
                                     elem_classes="consult-btn",
                                     min_width=200)
    
    # Output Section
    with gr.Row():
        with gr.Column():
            with gr.Column(elem_classes="medical-card"):
                gr.Markdown("## 📊 DIAGNOSTIC RESULTS")
                
                with gr.Tabs():
                    with gr.TabItem("📝 ANALYSIS REPORT"):
                        speech_text = gr.Textbox(
                            label="🎤 PATIENT SYMPTOMS SUMMARY",
                            placeholder="Your symptoms and description will appear here...",
                            lines=3,
                            show_copy_button=True
                        )
                        
                        doctor_response = gr.Textbox(
                            label="🩺 AI DOCTOR'S ANALYSIS",
                            placeholder="Professional medical analysis will appear here...",
                            lines=6,
                            show_copy_button=True
                        )
                    
                    with gr.TabItem("🔊 VOICE DIAGNOSIS"):
                        gr.Markdown("### 🔊 LISTEN TO DIAGNOSIS")
                        audio_output = gr.Audio(
                            label="🔊 DOCTOR'S VOICE RESPONSE",
                            autoplay=True,
                            show_download_button=True
                        )
                        gr.Markdown("*The AI doctor will read the diagnosis aloud for your convenience*")
    
    # Quick Tips Section
    with gr.Row():
        with gr.Accordion("💡 QUICK TIPS FOR BEST RESULTS", open=False):
            gr.Markdown("""
            ### 🎯 How to Get the Best Analysis:
            
            **For Internal Issues (Fever, Pain, etc.):**
            - Use the symptom selector above
            - Add brief audio description if needed
            - No image required for internal problems
            
            **For Visible Issues:**
            - Take clear, well-lit photos
            - Include something for scale (like a coin)
            - Take multiple angles if possible
            
            **General Tips:**
            - Be specific about symptom duration
            - Mention any existing conditions
            - Describe what makes symptoms better/worse
            """)
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center; color: #1a237e; font-weight: 600; padding: 20px;'>
    <p>🩺 BUILT WITH ADVANCED AI TECHNOLOGY FOR EDUCATIONAL PURPOSES</p>
    <p>⚡ POWERED BY GROQ AI • GRADIO • MEDICAL VISION ANALYSIS</p>
    <p>⚠️ NOT FOR ACTUAL MEDICAL DIAGNOSIS OR TREATMENT DECISIONS</p>
    </div>
    """)
    
    # Event Handlers
    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, common_symptoms],
        outputs=[speech_text, doctor_response, audio_output]
    )
    
    # Clear button
    clear_btn.click(
        fn=lambda: [None, None, [], "", "", None],
        inputs=[],
        outputs=[audio_input, image_input, common_symptoms, speech_text, doctor_response, audio_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        debug=True
    )
