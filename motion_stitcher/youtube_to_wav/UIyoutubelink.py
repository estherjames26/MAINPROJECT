import tkinter as tk
from tkinter import messagebox, filedialog
import os
import subprocess
import sys

# Ensure MAINPROJECT is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from motion_stitcher.main.generate import generate_choreography

def download_audio():
    url = url_entry.get().strip()
    num_dancers = dancer_var.get()

    if "youtu.be/" in url:
        url = url.split("?")[0]

    if not url:
        messagebox.showerror("Error", "Please enter a valid YouTube URL.")
        return

    try:
        python_310_path = r"C:\\Users\\kemij\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
        yt_script = os.path.join(os.path.dirname(__file__), "youtube_conv.py")

        if not os.path.exists(python_310_path):
            raise FileNotFoundError(f"Python 3.10 not found at: {python_310_path}")
        if not os.path.exists(yt_script):
            raise FileNotFoundError(f"youtube_conv.py not found at: {yt_script}")

        result = subprocess.run(
            [python_310_path, yt_script, url],
            capture_output=True, text=True, check=True
        )
        wav_file = result.stdout.strip()

        messagebox.showinfo("Success", f"Audio saved to:\n{wav_file}")

        choreo_path = generate_choreography(audio_path=wav_file, num_dancers=num_dancers, visualise_blender=True)
        if choreo_path:
            messagebox.showinfo("Choreography Generated", f"Output saved at:\n{choreo_path}")
        else:
            messagebox.showerror("Generation Failed", "Choreography generation failed.")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Download Failed", f"YouTube download failed:\n{e.stderr}")
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error:\n{e}")

def use_wav_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    num_dancers = dancer_var.get()

    if file_path:
        try:
            messagebox.showinfo("WAV Selected", f"Selected audio:\n{file_path}")
            choreo_path = generate_choreography(audio_path=file_path, num_dancers=num_dancers, visualise_blender=True)
            if choreo_path:
                messagebox.showinfo("Choreography Generated", f"Output saved at:\n{choreo_path}")
            else:
                messagebox.showerror("Generation Failed", "Choreography generation failed.")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error:\n{e}")

root = tk.Tk()
root.title("YouTube/WAV to Choreography")
root.geometry("400x300")

tk.Label(root, text="Enter YouTube URL:").pack(pady=10)
url_entry = tk.Entry(root, width=50)
url_entry.pack()

dancer_var = tk.IntVar(value=1)
tk.Label(root, text="Select number of dancers:").pack(pady=5)
tk.Radiobutton(root, text="1 dancer", variable=dancer_var, value=1).pack()
tk.Radiobutton(root, text="2 dancers", variable=dancer_var, value=2).pack()
tk.Radiobutton(root, text="3 dancers", variable=dancer_var, value=3).pack()

tk.Button(root, text="Convert YouTube to WAV", command=download_audio).pack(pady=10)
tk.Button(root, text="Use existing WAV file", command=use_wav_file).pack(pady=5)

root.mainloop()

# https://www.youtube.com/watch?v=8MsUTfJyqNo
