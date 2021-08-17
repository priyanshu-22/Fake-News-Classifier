from flask import Flask, request, flash, url_for
from flask.templating import render_template
from werkzeug.utils import redirect
from classifier import classifier

app = Flask(__name__, instance_relative_config=True)
app.secret_key = 'f3cfe9ed8fae309f02079dbf'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    article = request.form.get('article')

    if (len(article.split(' ')) <= 300):
        flash("The article must be more than 450 words long. ")
        return redirect(url_for('index'))  

    prediction = classifier.predict(article)

    if prediction == 0:
        label = 'real'
        color = 'success'
    else:
        label = 'fake'
        color = 'danger'

    return render_template('predict.html', article=article, label=label, color=color)