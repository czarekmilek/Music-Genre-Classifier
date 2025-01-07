# Music-Genre-Classifier

### Packages to install:

brew install ffmpeg (macOS)

### Command to download mp3 from YT playlist:

**REQUIRES** ffmpeg installed
yt-dlp -x --audio-format mp3 -o "classic_folder/%(title)s.%(ext)s" "https://www.youtube.com/watch?v=P2l0lbn5TVg&list=PL2788304DC59DBEB4&index=1"
