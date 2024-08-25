from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load data
data_fake = pd.read_csv('Fake.csv').sample(n=5000, random_state=1)
data_true = pd.read_csv('True.csv').sample(n=5000, random_state=1)

# Label the data
data_fake["class"] = 0
data_true["class"] = 1

# Combine the datasets and shuffle them
data_merge = pd.concat([data_fake, data_true], axis=0, ignore_index=True)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|\W|https?://\S+|www\.\S+|<.*?>+|\n|[%s]|\w*\d\w*' % re.escape(string.punctuation), '', text)
    return text

# Apply the preprocessing function
data['text'] = data['text'].apply(wordopt)

# Define features and labels
x = data['text']
y = data['class']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Vectorize the text with tuned parameters
vectorization = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train the models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_jobs=-1, random_state=0)

# Fit models
lr.fit(xv_train, y_train)
dt.fit(xv_train, y_train)
rf.fit(xv_train, y_train)

# Define output label function
def op_label(n):
    if n == 0:
        return "FAKE NEWS ALERT!!!"
    elif n == 1:
        return "NOT A FAKE NEWS APPROVED"
    return "UNKNOWN"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news = request.form['news']
        processed_news = wordopt(news)
        vectorized_news = vectorization.transform([processed_news])

        predictions = {
            'LR': op_label(lr.predict(vectorized_news)[0]),
            'DT': op_label(dt.predict(vectorized_news)[0]),
            'RF': op_label(rf.predict(vectorized_news)[0])
        }

        return render_template('index.html', predictions=predictions)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
