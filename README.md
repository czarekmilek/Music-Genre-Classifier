# Music-Genre-Classifier

## SETUP

### 1. Clone repository:

> git clone https://github.com/PT00/Music-Genre-Classifier.git
>
> cd Music-Genre-Classifier

### 2. Create VENV (Optional)

> python -m venv env
>
> Linux / macOS:
> source env/bin/activate
>
> Windows:
> env\Scripts\activate

### 3. Install Python packages:

> pip install -r requirements.txt

### 4. Install other packages:

> _For YouTube audio download_:
> brew install ffmpeg (macOS)

## Terminal commands

### Download mp3 from YT playlist:

**REQUIRES** ffmpeg installed

> yt-dlp -x --audio-format mp3 -o "classic_folder/%(title)s.%(ext)s" "https://www.youtube.com/watch?v=P2l0lbn5TVg&list=PL2788304DC59DBEB4&index=1"
