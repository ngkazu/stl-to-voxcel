@echo off
cd /d D:\+Projects\repos\stl-to-voxcel\src
python -m pytest tests/ -v --tb=short > D:\+Projects\repos\stl-to-voxcel\test_output.txt 2>&1
echo Done
