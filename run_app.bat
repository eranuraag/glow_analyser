@echo off
echo ========================================
echo Starting SOTA Skin Analysis Application
echo ========================================

REM Activate virtual environment
call venv\Scripts\activate

REM Check if requirements are installed
python -c "import flask, cv2, mediapipe" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Start the application
echo Starting Flask server...
python app.py

pause