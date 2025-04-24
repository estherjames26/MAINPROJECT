Setting up:
Downloading datasets used:
AIST:
motion: https://google.github.io/aistplusplus_dataset/download.html
Download the complete dataset however only keep the motion files and ignore_list.txt, and put the motion folder and ignore_list.txt into data\AIST
wav: https://aistdancedb.ongaaccel.jp/database_download/
Download Musical pieces only (wav) which is under the ZIP files category and add the folder to data\AIST

AIOZ:
Link for downloading whole database is at https://github.com/aioz-ai/AIOZ-GDANCE?tab=readme-ov-file
motions and musics folder should then be moved to data\AIOZ

Additional tools needed for installation:

FBX:
https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-3-2/fbx202032_fbxpythonsdk_win.exe

This link leads to the python SDK which the system uses, 2020.3.2 due to its compatibility for python 3.7. In order to use it after downloading, files in
C:\Program Files\Autodesk\FBX\FBX Python SDK\2020.3.2\lib\Python37_x64
should be copied into the site packages of the created python 3.7 virtual environment. Additionally, within fbxCommom.py, for generated fbx files to be visible within Blender, 
[if "ascii" in lDesc:] 
needs to be changed into 
[if "binary" in lDesc:].

ffmpeg:

installing requirements:
the system overall uses python 3.7 except for youtube_conv.py as due to version incompatibility with yt-dlp, it uses python 3.10 instead for that file exclusively


Download and extract ffmpeg-git-full.7z    .ver .sha256 which is under git master builds(direct link: https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z), and set ffmpeg_cmd in youtue_conv.py to the ffmpeg.exe file within the bin folder of the extracted ffmpeg folder.
 
Blender:
https://www.blender.org/ Blender needs to be installed and then its path to the exe file has to be configured to
Running:

Prequisites: database creation:
1. First, create the necessary directories and databases by running:
   ```
   python motion_stitcher/main/build_database.py
   ```
   This will process the AIST++ and AIOZ datasets, creating a general database and separate dancer-specific databases (for 1, 2, and 3 dancers).

2. Train the ensemble models by running:
   ```
   python motion_stitcher/main/ensemble.py
   ```
   This creates the ensemble models (RF+SVM+KNN) for choreography generation for each dancer count (1, 2, and 3 dancers).

3. To use the system with a GUI interface, run:
   ```
   python motion_stitcher/youtube_to_wav/UIyoutubelink.py
   ```
   This opens a UI where you can:
   - Enter a YouTube URL to download and convert to WAV, then generate choreography
   - Use an existing WAV file to generate choreography
   - Select the number of dancers (1, 2, or 3)
   - The system will then process the input and generate the choreography using the trained ensemble models.
   - The generated choreography will be saved in the specified output directory
   - The motion then will be shown within blender




