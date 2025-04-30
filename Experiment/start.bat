@echo off
setlocal enabledelayedexpansion
echo Starting Dolphin with Mario Bros Wii RL Agent...

REM Get the absolute path to the repository root directory
set REPO_ROOT=%~dp0..

REM Setup Python modules for Dolphin
echo Setting up Python modules for Dolphin...

REM Create directories for data and screenshots in the Experiment directory if they don't exist
if not exist "%~dp0data" mkdir "%~dp0data"
if not exist "%~dp0screenshots" mkdir "%~dp0screenshots"

REM We don't need to copy utils_func.py to Dolphin directory anymore
REM since we're setting PYTHONPATH to include the scripts directory

REM Dynamically copy Python packages from virtual environment to Dolphin's Python
if exist "%REPO_ROOT%\venv\Lib\site-packages" (
    echo Checking Python packages...

    REM Create a flag file to track if we've already copied packages
    set "FLAG_FILE=%~dp0dolphin\python-embed\packages_copied.flag"

    REM Check if utils_func.py has been modified since last copy
    set COPY_NEEDED=0
    if not exist "!FLAG_FILE!" set COPY_NEEDED=1

    REM Only copy packages if needed
    if !COPY_NEEDED!==1 (
        echo Copying Python packages from virtual environment...

        REM Get a list of all directories in the site-packages folder
        for /d %%D in ("%REPO_ROOT%\venv\Lib\site-packages\*") do (
            set "PACKAGE_NAME=%%~nxD"

            REM Skip dist-info and egg-info directories
            echo !PACKAGE_NAME! | findstr /i "dist-info egg-info" > nul
            if errorlevel 1 (
                if not exist "%~dp0dolphin\python-embed\!PACKAGE_NAME!" (
                    echo Copying !PACKAGE_NAME!...
                    mkdir "%~dp0dolphin\python-embed\!PACKAGE_NAME!" 2>nul
                    xcopy /E /I /Y "%%D" "%~dp0dolphin\python-embed\!PACKAGE_NAME!" > nul
                )
            )
        )

        REM Also copy any .py files in the root of site-packages
        for %%F in ("%REPO_ROOT%\venv\Lib\site-packages\*.py") do (
            if not exist "%~dp0dolphin\python-embed\%%~nxF" (
                echo Copying %%~nxF...
                copy /Y "%%F" "%~dp0dolphin\python-embed\" > nul
            )
        )

        REM Create or update the flag file
        echo Packages copied on %DATE% %TIME% > "!FLAG_FILE!"
        echo Packages copied successfully.
    ) else (
        echo Python packages already copied. Skipping...
    )
) else (
    echo Virtual environment not found. Some features may be disabled.
)

REM Set Python path to include scripts directory
set PYTHONPATH=%REPO_ROOT%\scripts;%PYTHONPATH%

echo Setup complete. Starting Dolphin...

REM Change to the dolphin directory and run Dolphin with the script
cd %~dp0dolphin
Dolphin.exe --script "%REPO_ROOT%\scripts\myscript.py" --no-python-subinterpreters