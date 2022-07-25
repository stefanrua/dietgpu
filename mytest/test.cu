#include <cstdio>
#include <ctime>

#include <cuda_runtime.h>

#include "../dietgpu/ans/GpuANSCodec.h"
#include "../dietgpu/utils/StackDeviceMemory.h"

#include "cudaCheck.h"

int n_print = 1;

void print() {
    printf("%d\n", n_print);
    n_print++;
}

int compress(const void* in, const uint32_t* insize, void* out, uint32_t* outsize) {
    //dietgpu::StackDeviceMemory res;
    // Compression configuration
    auto config = dietgpu::ANSCodecConfig();

    // Number of separate, independent compression problems
    uint32_t numInBatch;

    // Host array with addresses of device pointers comprising the input batch
    // to compress
    void* in_dgpu[1];
    // Host array with sizes of batch members
    uint32_t inSize[1];

    // Optional (can be null): region in device memory of size 256 words
    // containing pre-calculated symbol counts (histogram) of the data to be
    // compressed
    uint32_t* histogram_dev;

    // Host array with addresses of device pointers for the compressed output
    // arrays. Each out[i] must be a region of memory of size at least
    // getMaxCompressedSize(inSize[i])
    void* out_dgpu[1];
    // Device memory array of size numInBatch (optional)
    // Provides the size of actual used memory in each output compressed batch
    uint32_t* outSize_dev;

    // stream on the current device on which this runs
    cudaStream_t stream;

    uint32_t maxsize = dietgpu::getMaxCompressedSize(*insize);
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(in_dgpu, *insize));
    CUDA_CHECK(cudaMalloc(out_dgpu, maxsize));
    CUDA_CHECK(cudaMalloc(&outSize_dev, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(in_dgpu, in, *insize, cudaMemcpyHostToDevice));

    int device = 0;
    size_t allocPerDevice = maxsize;
    auto res = dietgpu::StackDeviceMemory(device, allocPerDevice);
    numInBatch = 1;
    *inSize = *insize;
    histogram_dev = nullptr;

    int t = clock();
    dietgpu::ansEncodeBatchPointer(
            res,
            config,
            numInBatch,
            (const void**)in_dgpu,
            inSize,
            histogram_dev,
            out_dgpu,
            outSize_dev,
            stream);
    t = clock() - t;

    CUDA_CHECK(cudaMemcpy(outsize, outSize_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out, *out_dgpu, *outsize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(*in_dgpu));
    CUDA_CHECK(cudaFree(*out_dgpu));
    CUDA_CHECK(cudaFree(outSize_dev));

    return t;
}

int main(int argc, char* argv[]) {
    char* ifname = NULL;
    char* ofname = NULL;
    FILE* fp;
    void* orig;
    uint32_t origsize;
    void* comp;
    uint32_t compsize;
    uint32_t maxsize;
    int t;

    switch (argc) {
        case 2:
            ifname = argv[1];
            break;
        case 3:
            ifname = argv[1];
            ofname = argv[2];
            break;
        default:
            printf("Usage: %s INFILE [OUTFILE]\n", argv[0]);
            return 1;
    }

    fp = fopen(ifname, "rb");
    fseek(fp, 0L, SEEK_END);
    origsize = ftell(fp);
    maxsize = dietgpu::getMaxCompressedSize(origsize);
    rewind(fp);
    orig = malloc(origsize);
    comp = malloc(maxsize);
    fread(orig, origsize, 1, fp);
    fclose(fp);

    t = compress(orig, &origsize, comp, &compsize);

    /* t = clock();
    compress(orig, &origsize, comp, &compsize);
    t = clock() - t; */

    if (ofname) {
        fp = fopen(ofname, "wb");
        fwrite(comp, compsize, 1, fp);
        fclose(fp);
        printf("wrote %s\n", ofname);
    }

    printf("insize: %u\n", origsize);
    printf("outsize: %u\n", compsize);
    printf("ratio: %f\n", (float)compsize/origsize);
    printf("throughput (comp): %f MB/s\n", (float)origsize/t * CLOCKS_PER_SEC/1000000);

    free(orig);
    free(comp);
}
