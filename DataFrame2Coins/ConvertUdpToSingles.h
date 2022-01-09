#pragma once
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "UAcquireDef.h"
#include "CoinPetPara.h"
template <unsigned blockSize>
inline __device__ unsigned intraWarpScan(volatile unsigned* scanTile, unsigned val) {
	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

	scanTile[index] = 0;              // 将前面一列置零
	index += min(blockSize, WARP_SIZE);
	scanTile[index] = val;

	if (blockSize >= 2)
	{
		scanTile[index] += scanTile[index - 1];
	}

	if (blockSize >= 4)
	{
		scanTile[index] += scanTile[index - 2];
	}
	if (blockSize >= 8)
	{
		scanTile[index] += scanTile[index - 4];
	}
	if (blockSize >= 16)
	{
		scanTile[index] += scanTile[index - 8];
	}
	if (blockSize >= 32)
	{
		scanTile[index] += scanTile[index - 16];
	}
	// 多个元素的值进行合并
	return scanTile[index] - val;
}


template <unsigned blockSize>
inline __device__ unsigned intraBlockScan(unsigned val) {
	__shared__ unsigned scanTile[blockSize * 2];                 // 这里要建立的共享缓存有多大？？？
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp


	unsigned warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();


	if (laneIdx == WARP_SIZE - 1)                 // 得到32个值的总和放在对应的warpIdx中
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();


	if (threadIdx.x < WARP_SIZE)                  // 仅用其中一个warp进行操作
	{
		scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
	}
	__syncthreads();


	return warpResult + scanTile[warpIdx] + val;

}

__device__ bool coinEnergyKernel(double energy, const CoinPetPara& mCoinPetPara) {
	bool isValid = energy >= mCoinPetPara.m_fMinEnergy && energy <= mCoinPetPara.m_fMaxEnergy;
	return isValid;
}



template <unsigned numThreads, unsigned elemsThread>
inline __device__ void calcDataBlockLength(unsigned& offset, unsigned& dataBlockLength, unsigned arrayLength)
{
	unsigned elemsPerThreadBlock = numThreads * elemsThread;            // 计算每个线程块要处理的数据量
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength = offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;       // 对最后一个线程块的特殊处理
}
template <unsigned threadsConvert, unsigned elemsConvert>
__global__ void convertUdpToSinglesKernel(DataFrameV2* src, SinglesStruct* dst, SinglesStruct* dstBuffer, unsigned arrayLength, const CoinPetPara mCoinPetPara, uint8* d_m_pPositionTable, float* d_m_pEnergyCorrFactor, bool mIsCoinEnergy, unsigned* coinEnergyLength) {

	unsigned offset, dataBlockLength;
	calcDataBlockLength<threadsConvert, elemsConvert>(offset, dataBlockLength, arrayLength);     //	找到每个线程块操作元素的偏移量和长度

	/* Position, Energy, Time corrections included */
	unsigned m_nChannelNum = mCoinPetPara.m_nChannelNum;
	//    unsigned moduleNumX = mCoinPetPara.m_nModuleNumX;
	unsigned moduleNumY = mCoinPetPara.m_nModuleNumY;
	//    unsigned moduleNumZ = mCoinPetPara.m_nModuleNumZ;
	//    unsigned blockNumX = mCoinPetPara.m_nBlockNumX;
	unsigned blockNumY = mCoinPetPara.m_nBlockNumY;
	unsigned blockNumZ = mCoinPetPara.m_nBlockNumZ;
	//    unsigned crystalNumX = mCoinPetPara.m_nCrystalNumX;
	unsigned crystalNumY = mCoinPetPara.m_nCrystalNumY;
	unsigned crystalNumZ = mCoinPetPara.m_nCrystalNumZ;
	unsigned positionSize = mCoinPetPara.m_nPositionSize;

	unsigned localTrue = 0;
	unsigned scanTrue = 0;
	const unsigned len = threadsConvert * elemsConvert;

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += threadsConvert) {
		/* Temporary structure to provide BDM and DU info */
		// 临时结构，提供BDM和DU信息  
		TempSinglesStruct temp;
		temp.globalBDMIndex = src[tx + offset].nBDM;
		temp.localDUIndex = src[tx + offset].nHeadAndDU & (0x0F);                   // 为什么要取与运算（截去前8个位，留下后面8个位）
		/* Time convertion, from unsigned char[8] to double */
		// 时间转换，从unsigned char[8]到double  
		uint64 nTimeTemp;
		nTimeTemp = src[tx + offset].nTime[0];
		for (unsigned i = 1; i <= 7; ++i) {
			nTimeTemp <<= 8;                                            // 将8个8位数据转换为一个64位数据
			nTimeTemp |= src[tx + offset].nTime[i];
		}
		temp.timevalue = (double)nTimeTemp;


		/* Position correction */
		// 位置校正（这里是怎么样的结构，在计算什么）
		uint32 originCrystalIndex = (d_m_pPositionTable + temp.globalBDMIndex * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nPositionSize * mCoinPetPara.m_nPositionSize +
			temp.localDUIndex * mCoinPetPara.m_nPositionSize * mCoinPetPara.m_nPositionSize)[src[tx + offset].X + src[tx + offset].Y * positionSize];
		uint32 localX = originCrystalIndex % crystalNumZ;
		uint32 localY = originCrystalIndex / crystalNumY;
		temp.localCrystalIndex = localX + (crystalNumY - 1 - localY) * crystalNumZ;


		/* Time correction */

		/* Energy convertion, from unsigned char[2] to float */
		uint32 nEnergyTemp;
		nEnergyTemp = (src[tx + offset].Energy[0] << 8 | src[tx + offset].Energy[1]);              // 将两个能量进行合并
		temp.energy = (float)nEnergyTemp;

		/* Up to different system structure ||Changeable|| */
		//uint32 nCrystalIdInRing = temp.globalBDMIndex % m_nChannelNum * m_nCrystalSize + (m_nCrystalSize - temp.localCrystalIndex / m_nCrystalSize -1);
		//uint32 nRingId = temp.localDUIndex % m_nDUNum * m_nCrystalSize + temp.localCrystalIndex % m_nCrystalSize;
		//uint32 nCrystalNumOneRing = m_nCrystalSize * m_nChannelNum;




		uint32 nCrystalIdInRing = temp.globalBDMIndex % (m_nChannelNum * moduleNumY) * blockNumY * crystalNumY + temp.localDUIndex / blockNumZ * crystalNumY + temp.localCrystalIndex / crystalNumZ;
		uint32 nRingId = temp.globalBDMIndex / (m_nChannelNum * moduleNumY) * blockNumZ * crystalNumZ + temp.localDUIndex % blockNumZ * crystalNumZ + temp.localCrystalIndex % crystalNumZ;
		uint32 nCrystalNumOneRing = crystalNumY * blockNumY * m_nChannelNum;


		dstBuffer[offset + tx].globalCrystalIndex = nCrystalIdInRing + nRingId * nCrystalNumOneRing;

		/* Energy correction */
		// 能量校正
		dstBuffer[offset + tx].energy = temp.energy * (d_m_pEnergyCorrFactor + temp.globalBDMIndex * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nCrystalSize * mCoinPetPara.m_nCrystalSize * 1000 +
			temp.localDUIndex * mCoinPetPara.m_nCrystalSize * mCoinPetPara.m_nCrystalSize * 1000 +
			temp.localCrystalIndex * 1000)[int(floor(temp.energy / 10))];

		dstBuffer[offset + tx].timevalue = temp.timevalue;

		// 这里要进行能量的筛选和合并
		// TODO: THE RIGHT PLACE FOR COIN ENERGY!

		// 计算每个线程操作的元素中能量满足要求的元素个数
		localTrue += coinEnergyKernel(dstBuffer[offset + tx].energy, mCoinPetPara);

	}

	// 针对每个线程块的操作不同
	// 得到这个线程块内，每个线程之前（包括该线程）的所有线程中为TRUE的元素个数
	scanTrue = intraBlockScan<threadsConvert>(localTrue);
	__syncthreads();


	__shared__ unsigned globalTrue;
	if (threadIdx.x == (threadsConvert - 1))										        // 每个线程块中的最后一个线程，此线程中scanLower存放的是本线程块中小于pivots的元素总数
	{
		globalTrue = atomicAdd(coinEnergyLength, scanTrue);                                // 原子操作，必须一个个进行
	}

	__syncthreads();

	unsigned indexTrue = globalTrue + scanTrue - localTrue;		// 每个线程中小于pivots要存入的起始位置

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += threadsConvert)                     // 这里完成合并操作
	{
		SinglesStruct temp = dstBuffer[offset + tx];   // 取出每一个要处理的元素
		if (coinEnergyKernel(temp.energy, mCoinPetPara))
		{
			dst[indexTrue++] = temp;
		}
	}
}

template <unsigned threadsConvert, unsigned elemsConvert>
__global__ void convertUdpToSinglesKernel(DataFrameV2* src, SinglesStruct* dst, unsigned arrayLength, const CoinPetPara mCoinPetPara, uint8* d_m_pPositionTable, float* d_m_pEnergyCorrFactor, bool mIsCoinEnergy, unsigned* coinEnergyLength) {

	unsigned offset, dataBlockLength;
	calcDataBlockLength<threadsConvert, elemsConvert>(offset, dataBlockLength, arrayLength);     //	找到每个线程块操作元素的偏移量和长度


	/* Position, Energy, Time corrections included */
	unsigned m_nChannelNum = mCoinPetPara.m_nChannelNum;
	//    unsigned moduleNumX = mCoinPetPara.m_nModuleNumX;
	unsigned moduleNumY = mCoinPetPara.m_nModuleNumY;
	//    unsigned moduleNumZ = mCoinPetPara.m_nModuleNumZ;
	//    unsigned blockNumX = mCoinPetPara.m_nBlockNumX;
	unsigned blockNumY = mCoinPetPara.m_nBlockNumY;
	unsigned blockNumZ = mCoinPetPara.m_nBlockNumZ;
	//    unsigned crystalNumX = mCoinPetPara.m_nCrystalNumX;
	unsigned crystalNumY = mCoinPetPara.m_nCrystalNumY;
	unsigned crystalNumZ = mCoinPetPara.m_nCrystalNumZ;
	unsigned positionSize = mCoinPetPara.m_nPositionSize;

	unsigned localTrue = 0;
	unsigned scanTrue = 0;
	const unsigned len = threadsConvert * elemsConvert;

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += threadsConvert) {
		/* Temporary structure to provide BDM and DU info */
		// 临时结构，提供BDM和DU信息  
		TempSinglesStruct temp;
		temp.globalBDMIndex = src[tx + offset].nBDM;
		temp.localDUIndex = src[tx + offset].nHeadAndDU & (0x0F);                   // 为什么要取与运算（截去前8个位，留下后面8个位）
		/* Time convertion, from unsigned char[8] to double */
		// 时间转换，从unsigned char[8]到double  
		uint64 nTimeTemp;
		nTimeTemp = src[tx + offset].nTime[0];
		for (unsigned i = 1; i <= 7; ++i) {
			nTimeTemp <<= 8;                                            // 将8个8位数据转换为一个64位数据
			nTimeTemp |= src[tx + offset].nTime[i];
		}
		temp.timevalue = (double)nTimeTemp;


		/* Position correction */
		// 位置校正（这里是怎么样的结构，在计算什么）
		uint32 originCrystalIndex = (d_m_pPositionTable + temp.globalBDMIndex * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nPositionSize * mCoinPetPara.m_nPositionSize +
			temp.localDUIndex * mCoinPetPara.m_nPositionSize * mCoinPetPara.m_nPositionSize)[src[tx + offset].X + src[tx + offset].Y * positionSize];
		uint32 localX = originCrystalIndex % crystalNumZ;
		uint32 localY = originCrystalIndex / crystalNumY;
		temp.localCrystalIndex = localX + (crystalNumY - 1 - localY) * crystalNumZ;


		/* Time correction */

		/* Energy convertion, from unsigned char[2] to float */
		uint32 nEnergyTemp;
		nEnergyTemp = (src[tx + offset].Energy[0] << 8 | src[tx + offset].Energy[1]);              // 将两个能量进行合并
		temp.energy = (float)nEnergyTemp;

		/* Up to different system structure ||Changeable|| */
		//uint32 nCrystalIdInRing = temp.globalBDMIndex % m_nChannelNum * m_nCrystalSize + (m_nCrystalSize - temp.localCrystalIndex / m_nCrystalSize -1);
		//uint32 nRingId = temp.localDUIndex % m_nDUNum * m_nCrystalSize + temp.localCrystalIndex % m_nCrystalSize;
		//uint32 nCrystalNumOneRing = m_nCrystalSize * m_nChannelNum;




		uint32 nCrystalIdInRing = temp.globalBDMIndex % (m_nChannelNum * moduleNumY) * blockNumY * crystalNumY + temp.localDUIndex / blockNumZ * crystalNumY + temp.localCrystalIndex / crystalNumZ;
		uint32 nRingId = temp.globalBDMIndex / (m_nChannelNum * moduleNumY) * blockNumZ * crystalNumZ + temp.localDUIndex % blockNumZ * crystalNumZ + temp.localCrystalIndex % crystalNumZ;
		uint32 nCrystalNumOneRing = crystalNumY * blockNumY * m_nChannelNum;


		dst[offset + tx].globalCrystalIndex = nCrystalIdInRing + nRingId * nCrystalNumOneRing;

		/* Energy correction */
		// 能量校正
		dst[offset + tx].energy = temp.energy * (d_m_pEnergyCorrFactor + temp.globalBDMIndex * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nCrystalSize * mCoinPetPara.m_nCrystalSize * 1000 +
			temp.localDUIndex * mCoinPetPara.m_nCrystalSize * mCoinPetPara.m_nCrystalSize * 1000 +
			temp.localCrystalIndex * 1000)[int(floor(temp.energy / 10))];

		dst[offset + tx].timevalue = temp.timevalue;

	}
	*coinEnergyLength = arrayLength;
}