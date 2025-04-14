from __future__ import unicode_literals
import yt_dlp
import sys
import os
import tempfile
import ffmpeg

output_dir = tempfile.mkdtemp()
intermediate_path = os.path.join(output_dir, 'output.m4a')
output_path = os.path.join(output_dir, 'output.wav')

ffmpeg_cmd = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'  # fallback to hardcoded path

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': intermediate_path,
    'quiet': True,
    'noplaylist': True,
    'postprocessors': []  # disable automatic ffmpeg processing
}

def download_from_url(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Convert to WAV using ffmpeg-python with hardcoded executable path
    ffmpeg.input(intermediate_path).output(output_path).run(overwrite_output=True, cmd=ffmpeg_cmd)

    print(output_path)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python youtube_conv.py <YouTube URL>", file=sys.stderr)
        sys.exit(1)
    download_from_url(args[0])
#python C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\youtube_to_wav\youtube_conv.py https://www.youtube.com/watch?v=8MsUTfJyqNo
