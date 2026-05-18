@echo off
setlocal enabledelayedexpansion
set ROOT=%~dp0

for %%F in ("%ROOT%*.txt") do (
    set NAME=%%~nF
    echo Processing !NAME!
    if exist "%%~fF" (
        type "%%~fF" >> "%TEMP%\betlang-fixture.log"
    )
)

if errorlevel 1 (
    echo command failed
    exit /b 1
)

endlocal
