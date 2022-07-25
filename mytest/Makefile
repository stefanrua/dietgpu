# remember this:
#	 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/user/srua/src/dietgpu/build/lib
test: test.cu
	nvcc -I.. -I../third_party/glog/build -I../third_party/glog/src -L../build/lib -lgpu_ans -ldietgpu_utils test.cu -o test
