@Echo OFF

REM Batch file to briefly run all OpenCL SDK samples 

REM Samples are run in 'NO GL' mode (--qatest switch)
REM This runs the app for one compute cycle with no GL calls before continuing

REM Also combines output files into a common file: oclSDK_QA.txt

REM Clear and init oclSDK_QA.txt file 

echo oclSDK_QA.bat Starting...
echo oclSDK_QA.bat Starting... >oclSDK_QA.txt

REM Record Windows Version and Local Date and Time 
ver
ver >>oclSDK_QA.txt
date /t 
date /t >>oclSDK_QA.txt
time /t 
time /t >>oclSDK_QA.txt
echo.
echo. >>oclSDK_QA.txt
echo ------------------------------------------------
echo ------------------------------------------------>>oclSDK_QA.txt
echo.
echo. >>oclSDK_QA.txt

REM Start executing individual Sample programs
REM and concatenating their log files into oclSDK_QA.txt

oclDeviceQuery.exe --qatest --noprompt
type oclDeviceQuery.txt >> oclSDK_QA.txt

oclBandwidthTest.exe --csv --qatest --noprompt
type oclBandwidthTest.txt >> oclSDK_QA.txt

oclBandwidthTest.exe --csv --memory=pinned --qatest --noprompt
type oclBandwidthTest.txt >> oclSDK_QA.txt

oclVectorAdd.exe --qatest --noprompt
type oclVectorAdd.txt >> oclSDK_QA.txt

oclDotProduct.exe --qatest --noprompt
type oclDotProduct.txt >> oclSDK_QA.txt

oclMatVecMul.exe --qatest --noprompt
type oclMatVecMul.txt >> oclSDK_QA.txt

oclSortingNetworks.exe --qatest --noprompt
type oclSortingNetworks.txt >> oclSDK_QA.txt

oclRadixSort.exe --qatest --noprompt
type oclRadixSort.txt >> oclSDK_QA.txt

oclBlackScholes.exe --qatest --noprompt
type oclBlackScholes.txt >> oclSDK_QA.txt

oclConvolutionSeparable.exe --qatest --noprompt
type oclConvolutionSeparable.txt >> oclSDK_QA.txt

oclDCT8x8.exe --qatest --noprompt
type oclDCT8x8.txt >> oclSDK_QA.txt

oclDXTCompression.exe --qatest --noprompt
type oclDXTCompression.txt >> oclSDK_QA.txt

oclFDTD3D.exe --qatest --noprompt
type oclFDTD3D.txt >> oclSDK_QA.txt

oclHiddenMarkovModel.exe --qatest --noprompt
type oclHiddenMarkovModel>> oclSDK_QA.txt

oclQuasirandomGenerator.exe --qatest --noprompt
type oclQuasirandomGenerator.txt >> oclSDK_QA.txt

oclMersenneTwister.exe --qatest --noprompt
type oclMersenneTwister.txt >> oclSDK_QA.txt

oclMatrixMul.exe --qatest --noprompt
type oclMatrixMul.txt >> oclSDK_QA.txt

oclScan.exe --qatest --noprompt
type oclScan.txt >> oclSDK_QA.txt

oclReduction.exe --qatest --noprompt
type oclReduction.txt >> oclSDK_QA.txt

oclTranspose.exe --qatest --noprompt
type oclTranspose.txt >> oclSDK_QA.txt

oclCopyComputeOverlap.exe  --qatest --noprompt --sizemult=4 --workgroupmult=2
type oclCopyComputeOverlap.txt >> oclSDK_QA.txt

oclSimpleMultiGPU.exe --qatest --noprompt
type oclSimpleMultiGPU.txt >> oclSDK_QA.txt

oclHistogram.exe --qatest --noprompt
type oclHistogram.txt >> oclSDK_QA.txt

oclSimpleGL.exe --qatest --noprompt
type oclSimpleGL.txt >> oclSDK_QA.txt

oclPostProcessGL.exe --qatest --noprompt
type oclPostProcessGL.txt >> oclSDK_QA.txt

oclSimpleTexture3d.exe --qatest --noprompt
type oclSimpleTexture3d.txt >> oclSDK_QA.txt

oclSimpleD3D9Texture.exe --qatest --noprompt
type oclSimpleD3D9Texture.txt >> oclSDK_QA.txt

oclSimpleD3D10Texture.exe --qatest --noprompt
type oclSimpleD3D10Texture.txt >> oclSDK_QA.txt

oclVolumeRender.exe --qatest --noprompt
type oclVolumeRender.txt >> oclSDK_QA.txt

oclBoxFilter.exe --qatest --noprompt
type oclBoxFilter.txt >> oclSDK_QA.txt

oclBoxFilter.exe --qatest --noprompt --lmem
type oclBoxFilter.txt >> oclSDK_QA.txt

oclRecursiveGaussian.exe --qatest --noprompt
type oclRecursiveGaussian.txt >> oclSDK_QA.txt

oclSobelFilter.exe --qatest --noprompt
type oclSobelFilter.txt >> oclSDK_QA.txt

oclMedianFilter.exe --qatest --noprompt
type oclMedianFilter.txt >> oclSDK_QA.txt

oclParticles.exe --qatest --noprompt
type oclParticles.txt >> oclSDK_QA.txt

oclNbody.exe --qatest --noprompt
type oclNbody.txt >> oclSDK_QA.txt

oclNbody.exe --double --qatest --noprompt
type oclNbody.txt >> oclSDK_QA.txt
