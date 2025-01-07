# Music-Genre-Classifier 🎶

## SETUP 🚀

### 1. Clone repository:

> git clone https://github.com/PT00/Music-Genre-Classifier.git
>
> cd Music-Genre-Classifier

### 2. Create VENV (Optional)

> python -m venv env

#### Linux / macOS:

> source env/bin/activate

#### Windows:

> env\Scripts\activate

### 3. Install Python packages:

> pip install -r requirements.txt

### 4. Install other packages:

#### _For YouTube audio download_:

> > brew install ffmpeg (macOS)

## Terminal commands

### Install FFmpeg 📦

To use `yt-dlp` for downloading and extracting audio, FFmpeg must be installed on your system. Follow the instructions for your operating system:

#### macOS (via Homebrew)

> brew install ffmpeg

#### Linux (Ubuntu/Debian-based)

> sudo apt update
> sudo apt install ffmpeg

#### Windows

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).
2. Extract the archive and add the `bin` folder to your system's PATH.

> yt-dlp -x --audio-format mp3 -o "classic_folder/%(title)s.%(ext)s" "PLAYLIST_LINK"
