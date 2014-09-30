@echo off
cd Release
start /b /wait nvprof -o DarkChannel.vp DarkChannel.exe

pause