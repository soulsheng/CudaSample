@echo off
cd Release
start /b /wait DarkChannel.exe

rundll32.exe %Systemroot\%System32\shimgvw.dll,ImageView_Fullscreen %cd%\out.bmp

pause