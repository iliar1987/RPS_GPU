#include "RPS.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define PI 3.1415926536f
#define PI2_3 2.0943951023931953f
#define PI4_3 4.1887902047863905f
#define PI2 6.283185307179586f
#define PI_3 1.0471975511965976f



#define GET_SCORE(ARR,X,Y) ((ARR)[(X) + (Y)*width])
#define GET_PIXEL(ARR,X,Y) ((float *)((ARR) + (Y)*pitch) + 4 * (X))


__device__ float3 convert_one_pixel_to_rgb_f(float3 pixel) {
	float r, g, b;
	float h, s, v;

	h = pixel.x;
	s = pixel.y;
	v = pixel.z;

	float f = h / PI_3;
	int hi = (int)floorf(f) % 6;
	float temp;
	f = modff(f,&temp);
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));

	switch (hi)
	{
	case 0:
		r = v;
		g = t;
		b = p;
		break;
	case 1:
		r = q;
		g = v;
		b = p;
		break;
	case 2:
		r = p;
		g = v;
		b = t;
		break;
	case 3:
		r = p;
		g = q;
		b = v;
		break;
	case 4:
		r = t;
		g = p;
		b = v;
		break;
	default:
		r = v;
		g = p;
		b = q;
		break;
	}


	return float3 { r, g, b };
}

__global__ void kern1(unsigned char* surface, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel;
	if (x >= width || y >= height) return;

	pixel = (float *)(surface + y*pitch) + 4 * x;


	float centerx = width * 0.5f;
	float centery = height * 0.5f;

	float fXDiff = (x - centerx);
	float fYDiff = (y - centery);

	float fDistance = sqrtf(fXDiff*fXDiff + fYDiff*fYDiff);
	float fAngle = atan2f(fYDiff, fXDiff);

	float fColAngle = t + fAngle;

	float v = __saturatef(1.0f - fDistance / width);
	
	float3 P = convert_one_pixel_to_rgb_f(float3{ fColAngle ,1.0f,v });

	/*pixel[0] = P.x;
	pixel[1] = P.y;
	pixel[2] = P.z;*/
	pixel[0] += 0.01;
	pixel[1] += 0.01;
	pixel[2] += 0.01;
	pixel[3] = 1.0f;
}

__global__ void K_CalcScores(const unsigned char* arrPixels,float* arrScore, int width, int height, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x ;
	int y = blockIdx.y*blockDim.y + threadIdx.y ;
	if (x >= width || y >= height) return;

	const float* globalPixelIn = GET_PIXEL(arrPixels,x, y);
	const float pixelIn[3] = { globalPixelIn[0],globalPixelIn[1],globalPixelIn[2] };

	const float* neighbors[4];
	neighbors[0] = (x == 0 ) ? GET_PIXEL(arrPixels,width-1, y) : GET_PIXEL(arrPixels, x-1, y);
	neighbors[1] = (x == width - 1) ? GET_PIXEL(arrPixels,0, y) : GET_PIXEL(arrPixels,x+1,y);
	neighbors[2] = (y == 0) ? GET_PIXEL(arrPixels,x, height-1) : GET_PIXEL(arrPixels,x,y-1);
	neighbors[3] = (y==height-1) ? GET_PIXEL(arrPixels,x, 0) : GET_PIXEL(arrPixels,x,y+1);
	
	float scoreOut = 0.0f;
	for (int i = 0; i < 4; ++i)
	{
		const float this_neighbor[3] = { neighbors[i][0] ,neighbors[i][1],neighbors[i][2] };
		scoreOut += this_neighbor[0] * pixelIn[1] - this_neighbor[1] * pixelIn[0]
			+ this_neighbor[1] * pixelIn[2] - this_neighbor[2] * pixelIn[1]
			+ this_neighbor[2] * pixelIn[0] - this_neighbor[0] * pixelIn[2];
	}
	
	GET_SCORE(arrScore,x,y) = scoreOut;
}


__global__ void K_MakeNextFrame(const unsigned char* arrPixels, const float* arrScore, unsigned char *arrFrameOut, int width, int height, size_t pitch, const float fNormPower, const float fDiffusionPower, const float fDiffusionCoeff)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const float* pixelIn = GET_PIXEL(arrPixels, x, y);
	float* pixelOut = GET_PIXEL(arrFrameOut, x, y);

	const float* neighbors[5];
	neighbors[0] = (x == 0) ? GET_PIXEL(arrPixels, width - 1, y) : GET_PIXEL(arrPixels, x - 1, y);
	neighbors[1] = (x == width - 1) ? GET_PIXEL(arrPixels, 0, y) : GET_PIXEL(arrPixels, x + 1, y);
	neighbors[2] = (y == 0) ? GET_PIXEL(arrPixels, x, height - 1) : GET_PIXEL(arrPixels, x, y - 1);
	neighbors[3] = (y == height - 1) ? GET_PIXEL(arrPixels, x, 0) : GET_PIXEL(arrPixels, x, y + 1);
	neighbors[4] = pixelIn;

	float neighborScores[4];
	neighborScores[0] = (x == 0) ? GET_SCORE(arrScore, width - 1, y) : GET_SCORE(arrScore, x - 1, y);
	neighborScores[1] = (x == width - 1) ? GET_SCORE(arrScore, 0, y) : GET_SCORE(arrScore, x + 1, y);
	neighborScores[2] = (y == 0) ? GET_SCORE(arrScore, x, height - 1) : GET_SCORE(arrScore, x, y - 1);
	neighborScores[3] = (y == height - 1) ? GET_SCORE(arrScore, x, 0) : GET_SCORE(arrScore, x, y + 1);
	
	//choose highest ranking neighbor
	float fHighestScore = GET_SCORE(arrScore,x,y);
	int iChosenNeighbor = 4;
	for (int i = 0; i < 4; ++i)
	{
		if (1.01f * fHighestScore < neighborScores[i])
		{
			fHighestScore = neighborScores[i];
			iChosenNeighbor = i;
		}
	}

	const float* pixelChosen = neighbors[iChosenNeighbor];
	
	float pixelResult[3];
	float fSum = 0;
	//diffuse
	for (int i = 0;i < 3;++i)
	{
		pixelResult[i] = powf(pixelChosen[i] * fDiffusionCoeff + pixelIn[i], fDiffusionPower);
		fSum += powf(pixelResult[i], fNormPower);
	}

	//normalize
	if (fSum != 0)
	{
		fSum = powf(fSum, 1.0f/ fNormPower);
		for (int i = 0;i < 3;++i)
		{
			pixelResult[i] /= fSum;
		}
	}

	//assign output
	for (int i = 0; i < 3; ++i)
	{
		pixelOut[i] = pixelResult[i];
	}
	pixelOut[3] = 1.0f;
}

RPSSim::~RPSSim()
{
	cudaFree(m_d_lastFrame);
	getLastCudaError("cudaFree (g_texture_2d) failed");

	cudaFree(m_d_thisFrame);
	getLastCudaError("cudaFree (g_texture_2d) failed");
}

RPSSim::RPSSim(const char *strInitStatePath)
{
	_ASSERT(0);
}

int RPSSim::RandomizeBuffer(float *d_buffer)
{
	size_t nSize = m_height * m_width * 4;
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen,
		CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
		1234ULL));
	CURAND_CALL(curandGenerateUniform(gen, d_buffer, nSize));
	return 0;
}

RPSSim::RPSSim(int width, int height)
{
	m_width = width;
	m_height = height;

	cudaMallocPitch(&m_d_lastFrame, &m_pitch, width * sizeof(float) * 4, height);
	getLastCudaError("cudaMallocPitch (g_texture_2d) failed");

	cudaMallocPitch(&m_d_thisFrame, &m_pitch, width * sizeof(float) * 4, height);
	getLastCudaError("cudaMallocPitch (g_texture_2d) failed");

	cudaMalloc(&m_d_arrScores, width * height * sizeof(float));
	getLastCudaError("cudaMalloc (g_texture_2d) failed");

	if ( RandomizeBuffer(m_d_thisFrame) )
		printf("Error randomizing\n");
}

void* RPSSim::MakeOneRPSFrame(float t, const float fNormPower, const float fDiffusionPower, const float fDiffusionCoeff)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((GetWidth() + Db.x - 1) / Db.x, (GetHeight() + Db.y - 1) / Db.y);

	float* temp = m_d_lastFrame;
	m_d_lastFrame = m_d_thisFrame;
	m_d_thisFrame = temp;

	//kern1 << <Dg, Db >> >((unsigned char *)surface, width, height, pitch, t);
	K_CalcScores << <Dg, Db >> >((unsigned char *)m_d_lastFrame,m_d_arrScores, GetWidth(), GetHeight(), GetPitch());
	if (error != cudaSuccess)
	{
		printf("K_CalcScores() failed to launch error = %d\n", error);
	}

	K_MakeNextFrame << <Dg, Db >> > ((unsigned char*)m_d_lastFrame, m_d_arrScores, (unsigned char*)m_d_thisFrame,
		GetWidth(), GetHeight(), GetPitch(), fNormPower, fDiffusionPower, fDiffusionCoeff);
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("K_MakeNextFrame failed to launch error = %d\n", error);
	}
	return m_d_thisFrame;
}
