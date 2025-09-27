# streamlit_app.py
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------
# Load GPT-Neo model once
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return generator

generator = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Subject Auto Content Generator")
st.write("Enter a subject and generate subtopics, content, and a summary.")

subject = st.text_input("Enter the subject:", "")

if st.button("Generate Content") and subject:
    st.info("Generating subtopics... This may take a few seconds.")

    # ------------------------------
    # Generate subtopics
    # ------------------------------
    subtopics_prompt = f"Generate 5 subtopics for the subject '{subject}':"
    subtopics_output = generator(
        subtopics_prompt,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']

    subtopics = [line.strip("- ").strip() for line in subtopics_output.split("\n") if line.strip()]
    if len(subtopics) < 5:
        subtopics = [f"Subtopic {i}" for i in range(1, 6)]

    st.subheader("Subtopics")
    for idx, stp in enumerate(subtopics, 1):
        st.write(f"{idx}. {stp}")

    # ------------------------------
    # Generate content for each subtopic
    # ------------------------------
    st.info("Generating content for each subtopic...")
    contents = {}
    for stp in subtopics:
        content_prompt = f"Write a short explanation (150 words) about '{stp}' in simple language."
        content_output = generator(
            content_prompt,
            max_length=300,
            do_sample=True,
            temperature=0.7
        )[0]['generated_text']
        content_text = content_output.replace(content_prompt, "").strip()
        contents[stp] = content_text

    st.subheader("Contents")
    for idx, (stp, content) in enumerate(contents.items(), 1):
        st.write(f"**{idx}. {stp}**")
        st.write(content)

    # ------------------------------
    # Generate overall summary
    # ------------------------------
    st.info("Generating summary...")
    summary_prompt = f"Summarize the subject '{subject}' based on these contents:\n"
    for stp, content in contents.items():
        summary_prompt += f"\n{stp}: {content}\n"

    summary_output = generator(
        summary_prompt,
        max_length=400,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']
    summary = summary_output.replace(summary_prompt, "").strip()

    st.subheader("Summary")
    st.write(summary)
