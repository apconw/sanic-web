@echo off
powershell -ExecutionPolicy RemoteSigned -File "%~dp0deploy_and_init.ps1"
pause