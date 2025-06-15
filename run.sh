file=$1
outName=$2

nvcc $file utils/utils.cpp -I . -o $outName;
./$outName;