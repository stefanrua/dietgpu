.PHONY: all clean

# remember this:
#	 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/user/srua/src/dietgpu/build/lib

all: test

clean:
	rm -f test

test: test.cu
	/usr/local/cuda/bin/nvcc -std=c++17 -O2 -g -lineinfo -I.. -I../third_party/glog/build -I../third_party/glog/src -L../build/lib -lgpu_ans -ldietgpu_utils test.cu -o test
