#pragma once
#include "Samples2Singles.h"
#include "ConvertUdpToSingles.h"
#include "device_launch_parameters.h"
#include "UAcquireDef.h"
#include "UTypeDef.h"
#include "cuda_runtime.h"
#include "device_functions.h"





inline __device__  int searchIndex(SinglesStruct* data, unsigned curIndex, unsigned arrayLength)
{
	unsigned next = curIndex + 1;
	while (next < arrayLength && data[next].timevalue - data[curIndex].timevalue <= TIMEWINDOW) { ++next; }
	return next == curIndex + 1 ? -1 : next - 1;
}

inline __device__ int leftIndex(SinglesStruct* data, unsigned curIndex) 
{
	int left = curIndex - 1;
	unsigned pos = curIndex;
	while (left >= 0 && data[curIndex].timevalue - data[left].timevalue <= TIMEWINDOW) { --curIndex; --left;}
	return curIndex == pos ? -1 : curIndex;
}

template<unsigned ThreadElems>
inline __device__ void myCalcDealLength(unsigned& offset, unsigned& dataBlockLength, unsigned arrayLength) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index * ThreadElems < arrayLength) {
		offset = index * ThreadElems;
		dataBlockLength = offset + ThreadElems <= arrayLength ? ThreadElems : arrayLength - offset;  //特殊处理最后一块
	}
	else {
		dataBlockLength = -1;
	}

}

inline __global__ void CorrectPosKernel(SinglesStruct* d_SingleData, CoinStruct* d_data, unsigned* newLength, unsigned* scanId, unsigned* globalLen, unsigned arrayLength, int* nextIndex) {
	unsigned offset1, dataBlockLength1;
    unsigned global = 0;
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index * THREAD_ELEM < arrayLength) {
		offset1 = index * THREAD_ELEM;
		dataBlockLength1 = offset1 + THREAD_ELEM <= arrayLength ? THREAD_ELEM : arrayLength - offset1;  //特殊处理最后一块
	}
	else {
		return;
	}
	for (int i = 0; i < blockIdx.x; ++i) {
		global += globalLen[i];
	}

	unsigned localBegin = global + scanId[index];
	for (unsigned tx = 0; tx < dataBlockLength1; ++tx) {
		if (nextIndex[tx + offset1] != -1) {
			d_data[localBegin].nCoinStruct[0] = d_SingleData[tx + offset1];
			d_data[localBegin++].nCoinStruct[1] = d_SingleData[tx + offset1 + 1];
		}
	}

}

inline __global__ void FillDataKernel(SinglesStruct* d_SingleData, CoinStruct* d_CoinData, int* nextIndex, unsigned arrayLength, unsigned* newLength, unsigned* scanId, unsigned* globalLen) {
	unsigned offset1, dataBlockLength1;
	unsigned local1 = 0, scan1 = 0;
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ unsigned global1;
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index * THREAD_ELEM < arrayLength) {
		offset1 = index * THREAD_ELEM;
		dataBlockLength1 = offset1 + THREAD_ELEM <= arrayLength ? THREAD_ELEM : arrayLength - offset1;  //特殊处理最后一块
	}
	else {
		return;
	}
	for (unsigned tx = 0; tx < dataBlockLength1; ++tx) {
		if (nextIndex[tx + offset1] != -1) {
			local1++;
		}
	}

	__syncthreads();
	scan1 = intraBlockScan<THREAD_NUM>(local1);

	__syncthreads();
	if (threadIdx.x == THREAD_NUM - 1 || offset1 + dataBlockLength1 == arrayLength) {
		atomicAdd(newLength, scan1);
		//globalPos[blockIdx.x] = atomicAdd(newLength, scan1);
		globalLen[blockIdx.x] = scan1;
	}
	__syncthreads();
	scanId[id] = scan1 - local1;
}
inline __global__ void SelectKernel( SinglesStruct* data,int* nextIndex, unsigned arrayLength) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_NUM, THREAD_ELEM>(offset, dataBlockLength, arrayLength);

	////找到最左端的起点
	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_NUM) {
		int pos;
		int left = leftIndex(data, tx + offset);
		if (left != -1) {
			pos = left;        //左边起点
		}
		else {
			pos = tx + offset;                //起点是自身
		}
		if (nextIndex[tx + offset] == -1 && pos != tx + offset) {
			while (pos <= offset + tx) {
				if ((nextIndex[pos] - pos) >= 2) {
					int begin = pos;
					int end = nextIndex[pos];
					for (; begin <= end; ++begin) {
						nextIndex[begin] = -1;
					}
					pos = begin;
				}
				else if ((nextIndex[pos] - pos) == 1) {
					if (data[pos].globalCrystalIndex == data[nextIndex[pos]].globalCrystalIndex) {
						nextIndex[pos] = -1;
					}
					nextIndex[++pos] = -1;   //++pos位置是事件对中的第二个
					++pos;
				}
				else if (nextIndex[pos] == -1) {
					++pos;
				}
			}
		}
	}
}
inline __global__ void CoinTimeKernel(SinglesStruct* data, int* nextIndex, unsigned arrayLength) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREAD_NUM, THREAD_ELEM>(offset, dataBlockLength, arrayLength);
	//每个线程找到满足时间窗口的最右边的index放到nextIndex数组中
	//每个线程找到距它最远的满足每两个元素时间窗口都小于timewindow
	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_NUM) {
		nextIndex[tx + offset] = searchIndex(data, offset + tx, arrayLength);
	}

}

template <unsigned stride>
inline __device__ int binarySearchExclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	while (indexStart <= indexEnd)
	{
		// Floor to multiplier of stride - needed for strides > 1
		int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ UINT_MAX);

		if (target.timevalue < dataArray[index].timevalue)
		{
			indexEnd = index - stride;
		}
		else
		{
			indexStart = index + stride;
		}
	}

	return indexStart;
}


inline __device__ int binarySearchExclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	return binarySearchExclusive<1>(dataArray, target, indexStart, indexEnd);
}

template <unsigned stride>
inline __device__ int binarySearchInclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	while (indexStart <= indexEnd)
	{
		// 小于该值的能被strike整除的最大数
		int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ UINT_MAX);   // 找到偏左的元素   

		if (target.timevalue <= dataArray[index].timevalue)
		{
			indexEnd = index - stride;
		}
		else
		{
			indexStart = index + stride;
		}
	}

	return indexStart;
}


inline __device__ int binarySearchInclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	return binarySearchInclusive<1>(dataArray, target, indexStart, indexEnd);
}

__global__ void addPaddingKernel(SinglesStruct* arrayPrimary, SinglesStruct* arrayBuffer, unsigned start, unsigned paddingLength, SinglesStruct tempVal)
{
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREADS_PADDING, ELEMS_PADDING>(offset, dataBlockLength, paddingLength);       // 计算每个block的偏移量和长度（这个函数很有用）
	offset += start;

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PADDING)
	{
		unsigned index = offset + tx;
		arrayPrimary[index] = tempVal;
		arrayBuffer[index] = tempVal;

	}
}

template <unsigned threadsMerge, unsigned elemsThreadMerge>
__global__ void mergeSortKernel(SinglesStruct* dataTable)
{
	extern __shared__ SinglesStruct mergeSortTile[];

	unsigned elemsPerThreadBlock = threadsMerge * elemsThreadMerge;
	SinglesStruct* globalDataTable = dataTable + blockIdx.x * elemsPerThreadBlock;        // 划分

	// 如果每个线程对两个以上的元素进行排序，则需要使用缓冲区数组 
	SinglesStruct* mergeTile = mergeSortTile;
	SinglesStruct* bufferTile = mergeTile + elemsPerThreadBlock;

	// 将数据从全局内存存入共享缓存区
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		mergeTile[tx] = globalDataTable[tx];
	}


	// 对共享缓存区中的数据进行排序
	for (unsigned stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
	{
		__syncthreads();

		// 如果可以做到一次性取出全部数据并写会，就不需要写入缓冲区中了，此时每个线程只能处理两个数据
		for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += threadsMerge)       // 线程是连续的
		{
			unsigned offsetSample = tx & (stride - 1);
			unsigned offsetBlock = 2 * (tx - offsetSample);

			// 从偶数和奇数块加载元素(块被合并)  
			SinglesStruct elemEven = mergeTile[offsetBlock + offsetSample];
			SinglesStruct elemOdd = mergeTile[offsetBlock + offsetSample + stride];            // 相当于一个线程处理两个元素

			// 计算偶数块中的元素在奇数块中的位置，反之亦然  
			unsigned rankOdd = binarySearchInclusive(
				mergeTile, elemEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
			);
			unsigned rankEven = binarySearchExclusive(
				mergeTile, elemOdd, offsetBlock, offsetBlock + stride - 1                 // 注意两者查找的位置不同
			);

			bufferTile[offsetSample + rankOdd - stride] = elemEven;
			bufferTile[offsetSample + rankEven] = elemOdd;
		}

		SinglesStruct* temp = mergeTile;        // 数组内容交换
		mergeTile = bufferTile;
		bufferTile = temp;
	}

	__syncthreads();
	// 将合并好的数据从共享内存存入全局内存
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		globalDataTable[tx] = mergeTile[tx];
	}
}


template <unsigned subBlockSize>
__global__ void generateRanksKernel(SinglesStruct* data, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize)
{
	unsigned subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;                                      // 每个有序数组划分成多少个子块
	unsigned subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;                                         // 两个合并的排序数组的子块总数

	// 找到要处理的端点值
	SinglesStruct sampleValue = data[blockIdx.x * (blockDim.x * subBlockSize) + threadIdx.x * subBlockSize];     // 注意一个线程查找一个端点
	unsigned rankSampleCurrent = blockIdx.x * blockDim.x + threadIdx.x;                                     // 端点号
	unsigned rankSampleOpposite;

	// 找到该端点号（子块）所在的序列以及它的相对应序列 
	unsigned indexBlockCurrent = rankSampleCurrent / subBlocksPerSortedBlock;                               // 端点号对应的序列
	unsigned indexBlockOpposite = indexBlockCurrent ^ 1;                                                    // 相对应要处理的序列

	// 找到这个端点值在相对应序列的哪个子块
	if (indexBlockCurrent % 2 == 0)
	{
		rankSampleOpposite = binarySearchInclusive<subBlockSize>(                              // 有步幅的二分查找
			data, sampleValue, indexBlockOpposite * sortedBlockSize,                                      // 查找端点在另一个序列中的位置
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite = (rankSampleOpposite - sortedBlockSize) / subBlockSize;                       // 相对位置
	}
	else
	{
		rankSampleOpposite = binarySearchExclusive<subBlockSize>(
			data, sampleValue, indexBlockOpposite * sortedBlockSize,
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite /= subBlockSize;
	}

	// 计算合并块内的样本索引 
	unsigned sortedIndex = rankSampleCurrent % subBlocksPerSortedBlock + rankSampleOpposite;         // 所在块编号

	// 计算样本在当前和相对排序块中的位置
	unsigned rankDataCurrent = (rankSampleCurrent * subBlockSize % sortedBlockSize) + 1;             // 在当前序列中的相对位置+1
	unsigned rankDataOpposite;

	// 计算相对排序块内的子块索引（这里缩小了查找范围）
	unsigned indexSubBlockOpposite = sortedIndex % subBlocksPerMergedBlock - rankSampleCurrent % subBlocksPerSortedBlock - 1;
	unsigned indexStart = indexBlockOpposite * sortedBlockSize + indexSubBlockOpposite * subBlockSize + 1;     // 不取端点值
	unsigned indexEnd = indexStart + subBlockSize - 2;

	if ((int)(indexStart - indexBlockOpposite * sortedBlockSize) >= 0)
	{
		if (indexBlockOpposite % 2 == 0)   // 本身是奇序列
		{
			rankDataOpposite = binarySearchExclusive(           // 二分查找位置
				data, sampleValue, indexStart, indexEnd               // 满足条件的后面一个位置（这样保证了两个序列中存在两个相同的数字时也可以满足）
			);
		}
		else                              // 本身是偶序列
		{
			rankDataOpposite = binarySearchInclusive(
				data, sampleValue, indexStart, indexEnd        // 满足条件的位置
			);
		}

		rankDataOpposite -= indexBlockOpposite * sortedBlockSize;
	}
	else
	{
		rankDataOpposite = 0;
	}

	// 对应的每个子块结束位置
	if ((rankSampleCurrent / subBlocksPerSortedBlock) % 2 == 0)    // 本身是偶序列
	{
		ranksEven[sortedIndex] = rankDataCurrent;              // 这里多加了1，保证至少有一个元素参与合并
		ranksOdd[sortedIndex] = rankDataOpposite;
	}
	else
	{
		ranksEven[sortedIndex] = rankDataOpposite;
		ranksOdd[sortedIndex] = rankDataCurrent;               // 这里多加了1
	}
}

//合并由rank数组决定的连续的偶数和奇数子块。
template <unsigned subBlockSize>
__global__ void mergeKernel(
	SinglesStruct* data, SinglesStruct* dataBuffer, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize
)
{
	__shared__ SinglesStruct tileEven[subBlockSize];           // 每个子块的元素个数不超512
	__shared__ SinglesStruct tileOdd[subBlockSize];            // 子块中奇子块和偶子块的元素个数均不超过256

	unsigned indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;               // 找到当前线程块对应的子块
	unsigned indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;                  // 该线程块操作的子块所在的序列的操作的起始位置

	// 索引相邻的偶数和奇数块，它们将被合并  
	unsigned indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
	unsigned offsetEven, offsetOdd;
	unsigned numElementsEven, numElementsOdd;

	// 读取偶数和奇数子块的START索引  
	// 每个线程块操作相同的起始和结束位置
	if (blockIdx.x > 0)
	{
		indexStartEven = ranksEven[indexRank - 1];
		indexStartOdd = ranksOdd[indexRank - 1];
	}
	else
	{
		indexStartEven = 0;
		indexStartOdd = 0;
	}
	// 读取偶数和奇数子块的END索引  
	if (blockIdx.x < gridDim.x - 1)
	{
		indexEndEven = ranksEven[indexRank];
		indexEndOdd = ranksOdd[indexRank];
	}
	else                           // 序列对应的最后一个线程块
	{
		indexEndEven = sortedBlockSize;   // 最后一个序列
		indexEndOdd = sortedBlockSize;
	}

	numElementsEven = indexEndEven - indexStartEven;    // 求出该线程块操作的元素个数
	numElementsOdd = indexEndOdd - indexStartOdd;

	// 从偶数有序子块中读取数据
	if (threadIdx.x < numElementsEven)
	{
		offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
		tileEven[threadIdx.x] = data[offsetEven];
	}
	// 从奇数有序子块中读取数据
	if (threadIdx.x < numElementsOdd)
	{
		offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
		tileOdd[threadIdx.x] = data[offsetOdd + sortedBlockSize];
	}

	__syncthreads();

	if (threadIdx.x < numElementsEven)
	{
		unsigned rankOdd = binarySearchInclusive(tileOdd, tileEven[threadIdx.x], 0, numElementsOdd - 1);
		rankOdd += indexStartOdd;
		// 求出在全局数组中的位置
		dataBuffer[offsetEven + rankOdd] = tileEven[threadIdx.x];
	}

	if (threadIdx.x < numElementsOdd)
	{
		unsigned rankEven = binarySearchExclusive(tileEven, tileOdd[threadIdx.x], 0, numElementsEven - 1);
		rankEven += indexStartEven;
		dataBuffer[offsetOdd + rankEven] = tileOdd[threadIdx.x];
	}
}
