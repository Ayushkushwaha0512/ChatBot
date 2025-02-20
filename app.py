from flask import Flask, render_template, request, jsonify
from transformers import pipeline 
import warnings


warnings.filterwarnings("ignore")


app = Flask(__name__)


chatbot = pipeline("text-generation", model="gpt2")


def respond_to_query(user_query):
    """
    Generates a response to the user's query using the LLM.
    """
    
    prompt = f"You are a financial assistant for WeCredit, a FinTech company. \
              Your goal is to provide accurate and helpful information about loans, \
              credit reports, interest rates, and other financial services. \
              Answer the following question in a concise and professional manner: \
              {user_query}"

    
    response = chatbot(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    
    user_query = request.form["user_query"]

    
    try:
        response = respond_to_query(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Sorry, I encountered an error. Please try again. Error: {e}"})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
