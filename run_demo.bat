@echo off
echo Starting Smart Waste Segregation Demo...
echo.
pip install streamlit ultralytics opencv-python pillow -q
echo.
streamlit run demo_app.py
pause
