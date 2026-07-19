# predict.spec
# -----------------------------------------------------------------------------
# Builds HandTalk-INDIA into a standalone desktop app (--onedir style build:
# one folder containing the .exe plus all its dependencies). onedir is used
# instead of onefile because TensorFlow/MediaPipe are large and onefile's
# "unpack to a temp dir on every launch" behaviour makes startup noticeably
# slower and is more prone to antivirus false-positives.
#
# Usage (from the repo root, after downloading the two model files — see
# README > Pretrained Models):
#     pip install pyinstaller pyinstaller-hooks-contrib
#     pyinstaller predict.spec
#
# Output: dist/HandTalk-INDIA/  (zip this whole folder for a GitHub Release)
# -----------------------------------------------------------------------------

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# MediaPipe ships its own .tflite / .binarypb model files as package data.
# PyInstaller's static import analysis can't see these, so they must be
# collected explicitly or the packaged app will crash with "file not found"
# errors from inside mediapipe at runtime.
mediapipe_datas = collect_data_files("mediapipe")

app_datas = [
    ("word/isl_final_model.keras", "word"),
    ("word/label_map.json", "word"),
    ("sentence/isl_sentence_model.keras", "sentence"),
    ("sentence/sentence_label_map.json", "sentence"),
]

a = Analysis(
    ["predict.py"],
    pathex=[],
    binaries=[],
    datas=mediapipe_datas + app_datas,
    hiddenimports=[
        "pyttsx3.drivers",
        "pyttsx3.drivers.sapi5",   # Windows TTS driver
        "pyttsx3.drivers.nsss",    # macOS TTS driver
        "pyttsx3.drivers.espeak",  # Linux TTS driver
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HandTalk-INDIA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,          # keep a console window open so users can see
                            # [WARN]/[TTS]/[ERROR] log lines from predict.py
    # icon="icon.ico",     # uncomment and add an .ico file to brand the exe
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HandTalk-INDIA",
)
