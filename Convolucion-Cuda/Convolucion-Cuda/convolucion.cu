#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <common/cpu_bitmap.h>
//para el blockIdx.x.
#include <device_launch_parameters.h>


#include <common/book.h>
#include <common/cpu_anim.h>
#include <common/gl_helper.h>
#include <common/gpu_anim.h>

#include <iostream>
#include <cstdlib>
#include <chrono>
#include "imageLoader.cpp"

using namespace std;
//using namespace cv;

//#include "imageLoader.cpp"

#define GRIDVAL 20.0 

__global__ void convolution(const byte* original, byte* cpu, const unsigned int width, const unsigned height, const int* __restrict__ mask) {
    int x= threadIdx.x + blockIdx.x * blockDim.x;
    int y= threadIdx.y + blockIdx.y * blockDim.y;

    float val = 0.0;
    int cont = 0;

   int limite = 1;

    if (x > (limite - 1) && y > (limite - 1) && (x <width-1) && (y <height-1) ) {
        for (int i = -limite; i <= limite;i++) {
            for (int j = -limite; j <= limite; j++) {
                if (i == 0) break;
                val += (mask[cont]*original[(y-i)*width+(x-j)]);
                cont += 1;
            }
        }
        cpu[y * width + x] = sqrt((val * val));
    }
}


__global__ void convolution_GPU7x7(const byte* original, byte* cpu, const unsigned int width, const unsigned height, const int* __restrict__ mask) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx = 0.0;
    int cont = 0;
    int limite = 3;
    if (x > (limite-1) && y > (limite - 1) && (x < width - 1) && (y < height - 1)) {
        for (int i = -limite; i <= limite; i++) {
            for (int j = -limite; j <= limite; j++) {
                if (i == 0) break;
                dx += (mask[cont] * original[(y - i) * width + (x - j)]);
                cont += 1;
            }
        }
        cpu[y * width + x] = sqrt(dx * dx);
    }
}

__global__ void convolution_GPU13x13(const byte* original, byte* cpu, const unsigned int width, const unsigned height, const int* __restrict__ mask) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx = 0.0;
    int cont = 0;
    int limite = 6;
    if (x > (limite - 1) && y > (limite - 1) && (x < width - 1) && (y < height - 1)) {
        for (int i = -limite; i <= limite; i++) {
            for (int j = -limite; j <= limite; j++) {
                if (i == 0) break;
                dx += (mask[cont] * original[(y - i) * width + (x - j)]);
                cont += 1;
            }
        }
        cpu[y * width + x] = sqrt((dx * dx));
    }
}

void convolution_cpu3x3(const byte* original, byte* cpu, const unsigned int width, const unsigned height, const int* __restrict__ mask) {
    int cont = 0, val = 0;

    int limite = 1;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int i = -limite; i <= limite; i++) {
                for (int j = -limite; j <= limite; j++) {
                    val += (mask[cont] * original[(y - i) * width + (x - j)]);
                    cont += 1;
                }
            }
            cont = 0;
            cpu[y * width + x] = sqrt((val * val));
            val = 0.0;
        }
    }
}

void convolution_cpu7x7(const byte* original, byte* cpu, const unsigned int width, const unsigned height, const int* __restrict__ mask) {
    
    int cont = 0, dx = 0;
    int limite = 3;
    for (int y = limite; y < height - 1; y++) {
        for (int x = limite; x < width - 1; x++) {
            for (int i = -limite; i <= limite; i++) {
                for (int j = -limite; j <= limite; j++) {
                    dx += (mask[cont] * original[(y-i) * width + (x-j)]);
                    cont += 1;
                }
            }
            cpu[y * width + x] = sqrt(dx * dx);
            cont = 0;
            dx = 0.0;
        }
    }
}

void convolution_cpu13x13(const byte* original, byte* cpu, const unsigned int width, const unsigned height, const int* __restrict__ mask) {

    int cont = 0, dx = 0;
    int limite = 6;
    for (int y = limite; y < height - 1; y++) {
        for (int x = limite; x < width - 1; x++) {
            for (int i = -limite; i <= limite; i++) {
                for (int j = -limite; j <= limite; j++) {
                    dx += (mask[cont] * original[(y - i) * width + (x - j)]);
                    cont += 1;
                }
            }
            cpu[y * width + x] = sqrt(dx * dx);
            cont = 0;
            dx = 0.0;
        }
    }
}
int main(int argc, char* argv[]) {

    int* deviceMask;

    //-------------MASCARA 3X3----------
    
    const int sizeMask = 6;

    int mask[sizeMask] = { 1,2,1,
                             -1,-2,-1};

    const int sizeMaskCPU =9 ;
    int maskCPU[sizeMaskCPU] = { 1,2,1,
                     0,0,0,
                     -1,-2,-1 };
     
     
    //-------------MASCARA 7X7----------

    /*const int sizeMask = 42;
    int mask[sizeMask] = {-1,-1,-1,1,1,1,
    -1,-2,-2,2,2,1,
    -1,-2,-3,3,2,1,
    -1,-2,-3,3,2,1,
    -1,-2,-3,3,2,1,
    -1,-2,-2,2,2,1,
    -1,-1,-1,1,1,1};

    const int sizeMaskCPU = 49;
    int maskCPU[sizeMaskCPU] = { -1,-1,-1,0,1,1,1,
    -1,-2,-2,0,2,2,1,
    -1,-2,-3,0,3,2,1,
    -1,-2,-3,0,3,2,1,
    -1,-2,-3,0,3,2,1,
    -1,-2,-2,0,2,2,1,
    -1,-1,-1,0,1,1,1 };*/

    

    //-------------MASCARA 13X13
    /*const int sizeMask = 156;

    int mask[sizeMask] = { 7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7 };

    const int sizeMaskCPU = 169;
    int maskCPU[sizeMaskCPU] = {
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    7,6,5,4,3,2,1,2,3,4,5,6,7,
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7,
    -7,-6,-5,-4,-3,-2,-1,-2,-3,-4,-5,-6,-7};*/

    if (argc != 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [gris.png]", argv[0]);
        return 1;
    }


   imgData original = loadImage(argv[1]);
   int width = original.width;
   int height = original.height;

   imgData gpu_imagen(new byte[width * height], width, height);
   imgData cpu_imagen(new byte[width * height], width, height);

   cout << "Dimensiones Imagen: "<<"X="<<width<<" Y="<<height<<endl;
   /*-----------CPU------------*/
   auto c = chrono::system_clock::now();
   convolution_cpu3x3(original.pixels, cpu_imagen.pixels, width, height, maskCPU);
   chrono::duration<double> time_cpu = chrono::system_clock::now() - c;
   printf("CPU tiempo ejecucion    = %*.4f msec\n", 5, 1000 * time_cpu.count());


   /*-----------GPU------------*/
   //asignar en la GPU para la imagen original, imagen resultante con las dimensiones respecctivas
   byte* gpu_original, *gpu_convolution;
   cudaMalloc((void**)&gpu_original,(width*height));
   cudaMalloc((void**)&gpu_convolution, (width * height));
   cudaMalloc((void**)&deviceMask, sizeMask * sizeof(int)); //tamanio de la mascara a pasar

   //Transferir memoria host a GPU  
   //matriz de convolucion, imagen resultante array con 0
   cudaMemcpy(gpu_original, original.pixels, (width * height), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_convolution,0, (width * height), cudaMemcpyHostToDevice);
   cudaMemcpy(deviceMask, mask, sizeMask * sizeof(int), cudaMemcpyHostToDevice);

   // dimensiones para la gpu, hilos por bloque y numero de bloques
   dim3 threads(GRIDVAL, GRIDVAL, 1);
   //numero bloques redondear al inmediato superior
   cout << "Numero bloques: " << ceil(width / GRIDVAL) << endl;
   dim3 blocks(ceil(width/GRIDVAL),ceil(height/GRIDVAL),1);

   c = chrono::system_clock::now();
   //ejecutar funcion
   convolution<< <blocks, threads >> > (gpu_original, gpu_convolution, width, height, deviceMask);

   chrono::duration<double> time_gpu = chrono::system_clock::now() - c;

   //retornar valorees
   cudaMemcpy(gpu_imagen.pixels,gpu_convolution,(width*height), cudaMemcpyDeviceToHost);

   //guardar imagen
   writeImage(argv[1], "gpu3x3", gpu_imagen);
   writeImage(argv[1], "cpu3x3", cpu_imagen);

   printf("CUDA tiempo de ejecucion   = %*.4f msec\n" ,5, 1000 * time_gpu.count());
   
   cudaFree(gpu_original); 
   cudaFree(gpu_convolution);
   cudaFree(deviceMask);

}





