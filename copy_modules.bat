@echo off
echo Copying required modules from virtual environment to Dolphin's Python...

REM Create directories if they don't exist
mkdir "Experiment\dolphin\python-embed\PIL" 2>nul
mkdir "Experiment\dolphin\python-embed\pygetwindow" 2>nul

REM Copy PIL (Pillow) module
echo Copying PIL module...
xcopy /E /I /Y "venv\Lib\site-packages\PIL" "Experiment\dolphin\python-embed\PIL"

REM Copy PyGetWindow module
echo Copying PyGetWindow module...
xcopy /E /I /Y "venv\Lib\site-packages\pygetwindow" "Experiment\dolphin\python-embed\pygetwindow"
copy /Y "venv\Lib\site-packages\PyGetWindow-*.dist-info\*" "Experiment\dolphin\python-embed\" 2>nul

REM Copy PyRect module (dependency of PyGetWindow)
echo Copying PyRect module...
xcopy /E /I /Y "venv\Lib\site-packages\pyrect" "Experiment\dolphin\python-embed\pyrect"
copy /Y "venv\Lib\site-packages\PyRect-*.dist-info\*" "Experiment\dolphin\python-embed\" 2>nul

echo Done!
