from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('video_play_index.html')

@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory('videos', filename)

if __name__ == '__main__':
    app.run(debug=True)