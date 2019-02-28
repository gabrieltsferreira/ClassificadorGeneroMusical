from flask import Flask, render_template, request
from lyricsProcessing import genre_Classifier, setup


app = Flask(__name__)
setup()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        input = request.form['lyrics']

        genre = genre_Classifier(input)

        return render_template('result.html', input=genre)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
