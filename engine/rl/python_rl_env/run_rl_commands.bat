@echo off
setlocal

REM Run this script from engine\rl\python_rl_env
REM Example:
REM   cd engine\rl\python_rl_env
REM   run_rl_commands.bat demo

REM ---------------------------
REM Edit these paths as needed
REM ---------------------------
set "SIMC=../../../out/build/x64-Debug/simc.exe"
set "PROFILE=../../../profiles/my_rl_profile.simc"
set "RUN_DIR=../../../training_runs/simc_ppo_YYYYMMDD_HHMMSS"
set "EVAL_ITERATIONS=2000"
set "EPISODES=3"

REM Optional: use Release binary instead
REM set "SIMC=../../../out/build/x64-Release/simc.exe"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=help"

if /I "%MODE%"=="demo" (
  uv run python rl_bridge.py --mode demo --simc "%SIMC%" --profile "%PROFILE%" --episodes %EPISODES%
  goto :eof
)

if /I "%MODE%"=="multi" (
  uv run python rl_bridge.py --mode multi --simc "%SIMC%" --profile "%PROFILE%"
  goto :eof
)

if /I "%MODE%"=="resume" (
  uv run python rl_bridge.py --mode multi --simc "%SIMC%" --profile "%PROFILE%" --resume "%RUN_DIR%"
  goto :eof
)

if /I "%MODE%"=="eval" (
  uv run python rl_bridge.py --mode eval --simc "%SIMC%" --profile "%PROFILE%" --model-dir "%RUN_DIR%" --iterations %EVAL_ITERATIONS% --output-dir "%RUN_DIR%/evaluation"
  goto :eof
)

if /I "%MODE%"=="single" (
  uv run python rl_bridge.py --mode single --simc "%SIMC%" --profile "%PROFILE%"
  goto :eof
)

if /I "%MODE%"=="all" (
  uv run python rl_bridge.py --mode demo --simc "%SIMC%" --profile "%PROFILE%" --episodes %EPISODES%
  uv run python rl_bridge.py --mode multi --simc "%SIMC%" --profile "%PROFILE%"
  uv run python rl_bridge.py --mode multi --simc "%SIMC%" --profile "%PROFILE%" --resume "%RUN_DIR%"
  uv run python rl_bridge.py --mode eval --simc "%SIMC%" --profile "%PROFILE%" --model-dir "%RUN_DIR%" --iterations %EVAL_ITERATIONS% --output-dir "%RUN_DIR%/evaluation"
  uv run python rl_bridge.py --mode single --simc "%SIMC%" --profile "%PROFILE%"
  goto :eof
)

echo Usage: run_rl_commands.bat ^<mode^>
echo.
echo Modes:
echo   demo    - run_demo
echo   multi   - run_multi_thread_sb3
echo   resume  - run_multi_thread_sb3 with --resume
echo   eval    - run_evaluation
echo   single  - run_single_thread_sb3
echo   all     - run all commands sequentially

endlocal

