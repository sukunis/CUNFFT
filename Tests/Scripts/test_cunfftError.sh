#m=4
cd build/
make distclean
cmake .. -DCUT_OFF=4 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=512 -DTHREAD_DIM_Y=512 -LAH
make #> makeLog.txt 2>&1 
make install

cd ../src/
./simpleTest -d 1 -N 12 -M 10000
./simpleTest -d 2 -N 6 6 -M 10000
./simpleTest -d 3 -N 4 4 4 -M 10000


#m=6
cd build/
make distclean
cmake .. -DCUT_OFF=6 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=512 -DTHREAD_DIM_Y=512 -LAH 
make #> makeLog.txt 2>&1 
make install 

cd ../src/
./simpleTest -d 1 -N 12 -M 10000
./simpleTest -d 2 -N 6 6 -M 10000
./simpleTest -d 3 -N 4 4 4 -M 10000

#m=8
cd ../build/
make distclean
cmake .. -DCUT_OFF=8 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=512 -DTHREAD_DIM_Y=512 -LAH
make #> makeLog.txt 2>&1 
make install 
cd ../src/
./simpleTest -d 1 -N 12 -M 10000
./simpleTest -d 2 -N 6 6 -M 10000
./simpleTest -d 3 -N 4 4 4 -M 10000

#m=10
cd ../build/
make distclean
cmake .. -DCUT_OFF=10 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=512 -DTHREAD_DIM_Y=512 -LAH
make #> makeLog.txt 2>&1 
make install 
cd ../src/
./simpleTest -d 1 -N 12 -M 10000
./simpleTest -d 2 -N 6 6 -M 10000
./simpleTest -d 3 -N 4 4 4 -M 10000

#m=12
cd ../build/
make distclean
cmake .. -DCUT_OFF=12 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=512 -DTHREAD_DIM_Y=512 -LAH
make #> makeLog.txt 2>&1 
make install 
cd ../src/
./simpleTest -d 1 -N 12 -M 10000
./simpleTest -d 2 -N 6 6 -M 10000
./simpleTest -d 3 -N 4 4 4 -M 10000


#m=14
cd ../build/
make distclean
cmake .. -DCUT_OFF=14 -DCOM_FG_PSI=ON -DLARGE_INPUT=ON -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=OFF -DMEASURED_TIMES=OFF -DPRINT_CONFIG=OFF -DDEBUG=OFF -DTHREAD_DIM_X=512 -DTHREAD_DIM_Y=512 -LAH
make #> makeLog.txt 2>&1 
make install 
cd ../src/
./simpleTest -d 1 -N 12 -M 10000
./simpleTest -d 2 -N 6 6 -M 10000
./simpleTest -d 3 -N 4 4 4 -M 10000

