from flask import Flask, render_template, request, jsonify
from transformers import pipeline 
import warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained LLM (GPT-2 in this case)
chatbot = pipeline("text-generation", model="gpt2")

# Define a function to generate responses
def respond_to_query(user_query):
    """
    Generates a response to the user's query using the LLM.
    """
    # Define a prompt to guide the chatbot's response
    prompt = f"You are a financial assistant for WeCredit, a FinTech company. \
              Your goal is to provide accurate and helpful information about loans, \
              credit reports, interest rates, and other financial services. \
              Answer the following question in a concise and professional manner: \
              {user_query}"

    # Generate a response using the LLM
    response = chatbot(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle chatbot queries
@app.route("/ask", methods=["POST"])
def ask():
    # Get the user's query from the form
    user_query = request.form["user_query"]

    # Generate a response using the chatbot
    try:
        response = respond_to_query(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Sorry, I encountered an error. Please try again. Error: {e}"})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)