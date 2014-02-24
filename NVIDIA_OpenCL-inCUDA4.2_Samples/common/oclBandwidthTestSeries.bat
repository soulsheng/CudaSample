@Echo OFF

REM Batch file to run oclBandwidthTest sample from OpenCL 10 times each in Paged and Pinned modes

REM Also combines output files into a common file: oclBandwidthTestSeries.txt

REM Clear and init oclBandwidthTestSeries.txt file 
echo oclBandwidthTestSeries.bat Starting...
echo oclBandwidthTestSeries.bat Starting... >oclBandwidthTestSeries.txt

REM Record Windows Version and Local Date and Time 
ver
ver >>oclBandwidthTestSeries.txt
date /t 
date /t >>oclBandwidthTestSeries.txt
time /t 
time /t >>oclBandwidthTestSeries.txt

REM Device Query
echo.
echo. >>oclBandwidthTestSeries.txt
echo -----------------------------------------------------------
echo ----------------------------------------------------------->>oclBandwidthTestSeries.txt
echo.
echo. >>oclBandwidthTestSeries.txt
oclDeviceQuery.exe --noprompt
type oclDeviceQuery.txt >> oclBandwidthTestSeries.txt

REM Paged bandwidth test
echo.
echo. >>oclBandwidthTestSeries.txt
echo -----------------------------------------------------------
echo ----------------------------------------------------------->>oclBandwidthTestSeries.txt
echo *** Start 10 cycles of Bandwidth Testing with Paged Memory
echo *** Start 10 cycles of Bandwidth Testing with Paged Memory >>oclBandwidthTestSeries.txt
echo -----------------------------------------------------------
echo ----------------------------------------------------------->>oclBandwidthTestSeries.txt
echo.
echo. >>oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt


REM Pinned bandwidth test
echo.
echo. >>oclBandwidthTestSeries.txt
echo -----------------------------------------------------------
echo ----------------------------------------------------------->>oclBandwidthTestSeries.txt
echo *** Start 10 cycles of Bandwidth Testing with Pinned Memory
echo *** Start 10 cycles of Bandwidth Testing with Pinned Memory >>oclBandwidthTestSeries.txt
echo -----------------------------------------------------------
echo ----------------------------------------------------------->>oclBandwidthTestSeries.txt
echo.
echo. >>oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt

oclBandwidthTest.exe --memory=pinned --noprompt
type oclBandwidthTest.txt >> oclBandwidthTestSeries.txt
