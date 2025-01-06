#!/usr/bin/env python3
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr

'''
    main_chatbot.py: A simple customer support AI Agent to assist users with basic questions about Thoughtful AI.
    The agent will use predefined, hardcoded responses to answer common questions.

    Author: Viji Kesavan
'''


# FAQ database. Storing both the original text and their embeddings in memory for now.
faq_json = {
    "questions": [
        {
            "question": "Hello",
            "answer": "Hello! How can I assist you?"
        },
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]
}


qn_list = [item['question'] for item in faq_json['questions']]
ans_list = [item['answer'] for item in faq_json['questions']]

# Create embeddings for faq (for now only questions are used for matching with user input)
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(qn_list)

# Returns the bot response for a user input
def faq_based_response(message, history):
    # Use cosine similarity to measure similarity between user input and faq embeddings
    query_emb = model.encode([message])
    qn_confidence = model.similarity(query_emb, faq_embeddings)
    argmax_conf = np.argmax(qn_confidence)
    print(message)
    print(qn_confidence)

    # For low confidence scores, reply saying I don't understand that.
    conf_threshold = 0.25
    if (qn_confidence[0][argmax_conf] < conf_threshold):
        return "Sorry, I don't understand that. Please let me know if you have any other questions regarding Thoughtful AI."

    # todo: Ask disambiguation questions when two top confidence values are close together
    return ans_list[argmax_conf]

# Chat Interface

greet_message = "Thank you for your interest in Thoughtful AI.<br> How can I assist you today?"
chatbot = gr.Chatbot(placeholder=greet_message)

gr.ChatInterface(
    fn=faq_based_response,
    type="messages",
    title="Thoughtful AI Chat Assistant",
    chatbot=chatbot
).launch(share=True)