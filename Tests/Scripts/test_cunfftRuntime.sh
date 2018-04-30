cd build/
make distclean
cmake .. -DCUT_OFF=4 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=128 -DTHREAD_DIM_Y=128 -LAH
make #> makeLog.txt 2>&1 
make install 
cd ../src/

./simpleTest -d 1 -N 4 -M 16
./simpleTest -d 1 -N 5 -M 32
./simpleTest -d 1 -N 6 -M 64
./simpleTest -d 1 -N 7 -M 128
./simpleTest -d 1 -N 8 -M 256
./simpleTest -d 1 -N 9 -M 512
./simpleTest -d 1 -N 10 -M 1024
./simpleTest -d 1 -N 11 -M 2048
./simpleTest -d 1 -N 12 -M 4096
./simpleTest -d 1 -N 13 -M 8192
./simpleTest -d 1 -N 14 -M 16384
./simpleTest -d 1 -N 15 -M 32768
./simpleTest -d 1 -N 16 -M 65536
./simpleTest -d 1 -N 17 -M 131072
./simpleTest -d 1 -N 18 -M 262144
./simpleTest -d 1 -N 19 -M 524288
./simpleTest -d 1 -N 20 -M 1048576
./simpleTest -d 1 -N 21 -M 2097152
./simpleTest -d 1 -N 22 -M 4194304

./simpleTest -d 2 -N 3 3 -M 64
./simpleTest -d 2 -N 4 4 -M 256
./simpleTest -d 2 -N 5 5 -M 1024
./simpleTest -d 2 -N 6 6 -M 4096
./simpleTest -d 2 -N 7 7 -M 16384
./simpleTest -d 2 -N 8 8 -M 65536
./simpleTest -d 2 -N 9 9 -M 262144
./simpleTest -d 2 -N 10 10 -M 1048576

./simpleTest -d 3 -N 3 3 3 -M 512
./simpleTest -d 3 -N 4 4 4 -M 4096
./simpleTest -d 3 -N 5 5 5 -M 32768
./simpleTest -d 3 -N 6 6 6 -M 262144

cd ../build/
make distclean
cmake .. -DCUT_OFF=2 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=128 -DTHREAD_DIM_Y=128 -LAH
make #> makeLog.txt 2>&1 
make install 
cd ../src/

./simpleTest -d 1 -N 4 -M 16
./simpleTest -d 1 -N 5 -M 32
./simpleTest -d 1 -N 6 -M 64
./simpleTest -d 1 -N 7 -M 128
./simpleTest -d 1 -N 8 -M 256
./simpleTest -d 1 -N 9 -M 512
./simpleTest -d 1 -N 10 -M 1024
./simpleTest -d 1 -N 11 -M 2048
./simpleTest -d 1 -N 12 -M 4096
./simpleTest -d 1 -N 13 -M 8192
./simpleTest -d 1 -N 14 -M 16384
./simpleTest -d 1 -N 15 -M 32768
./simpleTest -d 1 -N 16 -M 65536
./simpleTest -d 1 -N 17 -M 131072
./simpleTest -d 1 -N 18 -M 262144
./simpleTest -d 1 -N 19 -M 524288
./simpleTest -d 1 -N 20 -M 1048576
./simpleTest -d 1 -N 21 -M 2097152
./simpleTest -d 1 -N 22 -M 4194304

./simpleTest -d 2 -N 3 3 -M 64
./simpleTest -d 2 -N 4 4 -M 256
./simpleTest -d 2 -N 5 5 -M 1024
./simpleTest -d 2 -N 6 6 -M 4096
./simpleTest -d 2 -N 7 7 -M 16384
./simpleTest -d 2 -N 8 8 -M 65536
./simpleTest -d 2 -N 9 9 -M 262144
./simpleTest -d 2 -N 10 10 -M 1048576

./simpleTest -d 3 -N 3 3 3 -M 512
./simpleTest -d 3 -N 4 4 4 -M 4096
./simpleTest -d 3 -N 5 5 5 -M 32768
./simpleTest -d 3 -N 6 6 6 -M 262144


