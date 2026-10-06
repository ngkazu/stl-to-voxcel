@echo off
cd /d D:\+Projects\repos\stl-to-voxcel
python -m pytest tests/ -v --tb=short 2>&1
echo Exit code: %ERRORLEVEL%
pause
