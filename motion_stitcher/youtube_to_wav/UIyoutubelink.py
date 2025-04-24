import tkinter as tk
from tkinter import filedialog
import os
import subprocess
import sys
import traceback

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use the ensemble-based generator
from motion_stitcher.main.ensemble import generate_ensemble_choreography as generate_choreography

# Windows constant to suppress console windows
if os.name == 'nt':
    CREATE_NO_WINDOW = 0x08000000


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube/WAV to Choreography")
        self.geometry("600x400")

        # Input URL
        tk.Label(self, text="Enter YouTube URL:").pack(pady=(10,0))
        self.url_entry = tk.Entry(self, width=60)
        self.url_entry.pack(pady=(0,10))

        # Dancer options
        self.dancer_var = tk.IntVar(value=1)
        tk.Label(self, text="Select number of dancers:").pack()
        frame = tk.Frame(self)
        tk.Radiobutton(frame, text="1 dancer", variable=self.dancer_var, value=1).pack(side='left', padx=5)
        tk.Radiobutton(frame, text="2 dancers", variable=self.dancer_var, value=2).pack(side='left', padx=5)
        tk.Radiobutton(frame, text="3 dancers", variable=self.dancer_var, value=3).pack(side='left', padx=5)
        frame.pack(pady=(0,10))

        # Buttons
        btn_frame = tk.Frame(self)
        tk.Button(btn_frame, text="Convert YouTube to WAV", command=self.download_audio).pack(side='left', padx=10)
        tk.Button(btn_frame, text="Use existing WAV file", command=self.use_wav_file).pack(side='left', padx=10)
        btn_frame.pack(pady=(0,10))

        # Status output
        tk.Label(self, text="Status:").pack(anchor='w', padx=10)
        self.status_text = tk.Text(self, height=10, wrap='word', state='disabled')
        self.status_text.pack(fill='both', expand=True, padx=10, pady=(0,10))

    def log(self, message: str):
        self.status_text.configure(state='normal')
        self.status_text.insert('end', message + '\n')
        self.status_text.see('end')
        self.status_text.configure(state='disabled')

    def download_audio(self):
        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', 'end')
        self.status_text.configure(state='disabled')
        url = self.url_entry.get().strip()
        dancers = self.dancer_var.get()
        self.log("Starting YouTube download...")

        if not url:
            self.log("Error: No URL provided.")
            return

        yt_script = os.path.join(os.path.dirname(__file__), "youtube_conv.py")
        python_310 = r"C:\Users\kemij\AppData\Local\Programs\Python\Python310\python.exe"
        cmd = [python_310, yt_script, url]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                creationflags=CREATE_NO_WINDOW
            )
            wav = res.stdout.strip().splitlines()[-1]
            self.log(f"Downloaded and converted to WAV: {wav}")

            self.log("Generating choreography...")
            pkl = generate_choreography(audio_path=wav, num_dancers=dancers, target_duration=900)
            if not pkl:
                self.log("Error: Choreography generation failed.")
                return
            self.log(f"Choreography saved: {pkl}")

            # launch Blender conversion
            blender_script = os.path.join(
                project_root, "motion_stitcher", "VISUALISATIONSMPLFBX", "convert_stitching.py"
            )
            subprocess.Popen(
                [sys.executable, blender_script, "--pkl", pkl],
                creationflags=CREATE_NO_WINDOW
            )
            self.log("Launching Blender for visualisation.")

        except subprocess.CalledProcessError as e:
            self.log(f"Download/conversion error: {e.stderr}")
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"Unexpected error: {str(e)}")
            self.log(tb)

    def use_wav_file(self):
        wav = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        dancers = self.dancer_var.get()
        if not wav:
            return
        self.log(f"Selected WAV: {wav}")
        try:
            self.log("Generating choreography...")
            target_duration = 900  # or any other value suitable for your choreography
            pkl = generate_choreography(audio_path=wav, num_dancers=dancers, target_duration=target_duration)
            if not pkl:
                self.log("Error: Choreography generation failed.")
                return
            self.log(f"Choreography saved: {pkl}")

            blender_script = os.path.join(
                project_root, "motion_stitcher", "VISUALISATIONSMPLFBX", "convert_stitching.py"
            )
            subprocess.Popen(
                [sys.executable, blender_script, "--pkl", pkl],
                creationflags=CREATE_NO_WINDOW
            )
            self.log("Launched Blender for visualisation.")
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"Unexpected error: {str(e)}")
            self.log(tb)

if __name__ == '__main__':
    App().mainloop()
