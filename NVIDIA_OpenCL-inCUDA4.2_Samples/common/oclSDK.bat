@Echo OFF

REM Batch file to briefly run all OpenCL SDK samples 

REM Samples are run in 'tour' mode (--noprompt switch)
REM This runs the full app with GL (if appropriate) for a finite period before continuing
REM It runs the last sample indefinitely until a manual
 escape/quit


REM Also combines output files into a common file: oclSDK.txt

REM Clear and init oclSDK.txt file 
echo oclSDK.bat Starting...
echo oclSDK.bat Starting... >oclSDK.txt

REM Record Windows Version and Local Date and Time 
ver
ver >>oclSDK.txt
date /t 
date /t >>oclSDK.txt
time /t 
time /t >>oclSDK.txt
echo.
echo. >>oclSDK.txt
echo ------------------------------------------------
echo ------------------------------------------------>>oclSDK.txt
echo.
echo. >>oclSDK.txt

REM Start executing individual Sample programs
REM and concatenating their log files into oclSDK.txt

oclDeviceQuery.exe --noprompt
type oclDeviceQuery.txt >> oclSDK.txt

oclBandwidthTest.exe --noprompt --csv
type oclBandwidthTest.txt >> oclSDK.txt

oclBandwidthTest.exe --memory=pinned --noprompt --csv
type oclBandwidthTest.txt >> oclSDK.txt

oclVectorAdd.exe --noprompt
type oclVectorAdd.txt >> oclSDK.txt

oclDotProduct.exe --noprompt
type oclDotProduct.txt >> oclSDK.txt

oclMatVecMul.exe --noprompt
type oclMatVecMul.txt >> oclSDK.txt

oclSortingNetworks.exe --noprompt
type oclSortingNetworks.txt >> oclSDK.txt

oclRadixSort.exe --noprompt
type oclRadixSort.txt >> oclSDK.txt

oclBlackScholes.exe --noprompt
type oclBlackScholes.txt >> oclSDK.txt

oclConvolutionSeparable.exe --noprompt
type oclConvolutionSeparable.txt >> oclSDK.txt

oclDCT8x8.exe --noprompt
type oclDCT8x8.txt >> oclSDK.txt

oclDXTCompression.exe --noprompt
type oclDXTCompression.txt >> oclSDK.txt

oclFDTD3D.exe --noprompt
type oclFDTD3D.txt >> oclSDK.txt

oclHiddenMarkovModel.exe --noprompt
type oclHiddenMarkovModel>> oclSDK.txt

oclQuasirandomGenerator.exe --noprompt
type oclQuasirandomGenerator.txt >> oclSDK.txt

oclMersenneTwister.exe --noprompt
type oclMersenneTwister.txt >> oclSDK.txt

oclMatrixMul.exe --noprompt
type oclMatrixMul.txt >> oclSDK.txt

oclScan.exe --noprompt
type oclScan.txt >> oclSDK.txt

oclReduction.exe --noprompt
type oclReduction.txt >> oclSDK.txt

oclTranspose.exe --noprompt
type oclTranspose.txt >> oclSDK.txt

oclCopyComputeOverlap.exe --noprompt --sizemult=4 --workgroupmult=2
type oclCopyComputeOverlap.txt >> oclSDK.txt

oclSimpleMultiGPU.exe --noprompt
type oclSimpleMultiGPU.txt >> oclSDK.txt

oclHistogram.exe --noprompt
type oclHistogram.txt >> oclSDK.txt

oclSimpleGL.exe --noprompt
type oclSimpleGL.txt >> oclSDK.txt

oclPostProcessGL.exe --noprompt
type oclPostProcessGL.txt >> oclSDK.txt

oclSimpleTexture3d.exe --noprompt
type oclSimpleTexture3d.txt >> oclSDK.txt

oclSimpleD3D9Texture.exe --qatest --noprompt
type oclSimpleD3D9Texture.txt >> oclSDK.txt

oclSimpleD3D10Texture.exe --qatest --noprompt
type oclSimpleD3D10Texture.txt >> oclSDK.txt

oclVolumeRender.exe --noprompt
type oclVolumeRender.txt >> oclSDK.txt

oclBoxFilter.exe --noprompt
type oclBoxFilter.txt >> oclSDK.txt

oclBoxFilter.exe --noprompt --lmem
type oclBoxFilter.txt >> oclSDK.txt

oclRecursiveGaussian.exe --noprompt
type oclRecursiveGaussian.txt >> oclSDK.txt

oclSobelFilter.exe --noprompt
type oclSobelFilter.txt >> oclSDK.txt

oclMedianFilter.exe --noprompt
type oclMedianFilter.txt >> oclSDK.txt

oclParticles.exe --noprompt
type oclParticles.txt >> oclSDK.txt

oclNbody.exe --noprompt
type oclNbody.txt >> oclSDK.txt

oclNbody.exe --double --noprompt
type oclNbody.txt >> oclSDK.txt
