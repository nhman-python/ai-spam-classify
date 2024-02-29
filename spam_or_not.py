import sys

import joblib

try:
    classifier = joblib.load('spam_model.pkl')
    vectorized = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Please train the model and save it before using it.")
    sys.exit(1)


user_input = input("Enter text to check for spam: ")


try:
    user_input_vectorized = vectorized.transform([user_input])
except ValueError:
    print(
        "Error: The input text could not be processed by the vectorizer. Please check the input for any invalid "
        "characters.")
    sys.exit(1)


prediction = classifier.predict(user_input_vectorized)


if prediction == 0:
    print("The text is not spam.")
else:
    print("The text is spam.")
