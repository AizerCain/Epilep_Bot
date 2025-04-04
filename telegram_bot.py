import os
import numpy as np
import tensorflow as tf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the trained model and preprocessing objects
model = tf.keras.models.load_model('epilepsy_model_final.keras')
tokenizer = joblib.load('tokenizer.pkl')
le = joblib.load('label_encoder.pkl')

# Risk level descriptions
risk_descriptions = {
    'High Risk': "⚠️ This is a HIGH RISK question about epilepsy. Please consult a medical professional immediately if this is an emergency.",
    'Moderate Risk': "ℹ️ This is a MODERATE RISK question about epilepsy. While not immediately dangerous, you should consult a healthcare provider for proper guidance.",
    'Low Risk': "ℹ️ This is a LOW RISK question about epilepsy. You may find helpful information in our resources, but always verify with a professional."
}

# Bot credentials
TOKEN = '7673369890:AAGZdt7AwngI0B5_y9sO92kZy7-qZF4zX9I'  # Set this in your environment variables
BOT_USERNAME = '@Epil6565Bot'  # Replace with your bot's username

# Text preprocessing function (must match training preprocessing)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to classify user input
def classify_message(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=25)
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    risk_level = le.inverse_transform(predicted_class)[0]
    
    return risk_level, prediction[0]

# Telegram bot handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """🤖 *Epilepsy Risk Assessment Bot*
    
I can help assess the risk level of questions about epilepsy. Just type your question and I'll analyze it.

*Examples:*
- What are epilepsy symptoms? (High Risk)
- What should I do during a seizure? (Moderate Risk)
- Can food affect seizures? (Low Risk)

*Disclaimer:* This bot provides risk assessment only, not medical advice. Always consult a healthcare professional for medical concerns.
"""
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_msg = """ℹ️ *Help*
    
This bot classifies epilepsy-related questions into risk categories:
- *High Risk:* Symptoms, emergencies (consult doctor immediately)
- *Moderate Risk:* Treatment, first aid (seek professional advice)
- *Low Risk:* General information, myths (educational purposes)

Just type your question about epilepsy and I'll assess its risk level.
"""
    await update.message.reply_text(help_msg, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type = update.message.chat.type
    text = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            text = text.replace(BOT_USERNAME, '').strip()
        else:
            return

    # Classify the message
    risk_level, confidence = classify_message(text)
    confidence_percent = round(max(confidence) * 100, 1)
    
    # Prepare response
    response = f"{risk_descriptions[risk_level]}\n\n"
    response += f"*Confidence:* {confidence_percent}%\n"
    response += "*Original question:* " + text
    
    await update.message.reply_text(response, parse_mode='Markdown')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')
    await update.message.reply_text("An error occurred. Please try again later.")

if __name__ == '__main__':
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error_handler)

    print('Polling...')
    app.run_polling(poll_interval=3)