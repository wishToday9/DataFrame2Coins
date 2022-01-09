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
		dataBlockLength = offset + ThreadElems <= arrayLength ? ThreadElems : arrayLength - offset;  //���⴦�����һ��
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
		dataBlockLength1 = offset1 + THREAD_ELEM <= arrayLength ? THREAD_ELEM : arrayLength - offset1;  //���⴦�����һ��
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
		dataBlockLength1 = offset1 + THREAD_ELEM <= arrayLength ? THREAD_ELEM : arrayLength - offset1;  //���⴦�����һ��
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

	////�ҵ�����˵����
	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREAD_NUM) {
		int pos;
		int left = leftIndex(data, tx + offset);
		if (left != -1) {
			pos = left;        //������
		}
		else {
			pos = tx + offset;                //���������
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
					nextIndex[++pos] = -1;   //++posλ�����¼����еĵڶ���
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
	//ÿ���߳��ҵ�����ʱ�䴰�ڵ����ұߵ�index�ŵ�nextIndex������
	//ÿ���߳��ҵ�������Զ������ÿ����Ԫ��ʱ�䴰�ڶ�С��timewindow
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
		// С�ڸ�ֵ���ܱ�strike�����������
		int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ UINT_MAX);   // �ҵ�ƫ���Ԫ��   

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
	calcDataBlockLength<THREADS_PADDING, ELEMS_PADDING>(offset, dataBlockLength, paddingLength);       // ����ÿ��block��ƫ�����ͳ��ȣ�������������ã�
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
	SinglesStruct* globalDataTable = dataTable + blockIdx.x * elemsPerThreadBlock;        // ����

	// ���ÿ���̶߳��������ϵ�Ԫ�ؽ�����������Ҫʹ�û��������� 
	SinglesStruct* mergeTile = mergeSortTile;
	SinglesStruct* bufferTile = mergeTile + elemsPerThreadBlock;

	// �����ݴ�ȫ���ڴ���빲������
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		mergeTile[tx] = globalDataTable[tx];
	}


	// �Թ��������е����ݽ�������
	for (unsigned stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
	{
		__syncthreads();

		// �����������һ����ȡ��ȫ�����ݲ�д�ᣬ�Ͳ���Ҫд�뻺�������ˣ���ʱÿ���߳�ֻ�ܴ�����������
		for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += threadsMerge)       // �߳���������
		{
			unsigned offsetSample = tx & (stride - 1);
			unsigned offsetBlock = 2 * (tx - offsetSample);

			// ��ż�������������Ԫ��(�鱻�ϲ�)  
			SinglesStruct elemEven = mergeTile[offsetBlock + offsetSample];
			SinglesStruct elemOdd = mergeTile[offsetBlock + offsetSample + stride];            // �൱��һ���̴߳�������Ԫ��

			// ����ż�����е�Ԫ�����������е�λ�ã���֮��Ȼ  
			unsigned rankOdd = binarySearchInclusive(
				mergeTile, elemEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
			);
			unsigned rankEven = binarySearchExclusive(
				mergeTile, elemOdd, offsetBlock, offsetBlock + stride - 1                 // ע�����߲��ҵ�λ�ò�ͬ
			);

			bufferTile[offsetSample + rankOdd - stride] = elemEven;
			bufferTile[offsetSample + rankEven] = elemOdd;
		}

		SinglesStruct* temp = mergeTile;        // �������ݽ���
		mergeTile = bufferTile;
		bufferTile = temp;
	}

	__syncthreads();
	// ���ϲ��õ����ݴӹ����ڴ����ȫ���ڴ�
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		globalDataTable[tx] = mergeTile[tx];
	}
}


template <unsigned subBlockSize>
__global__ void generateRanksKernel(SinglesStruct* data, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize)
{
	unsigned subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;                                      // ÿ���������黮�ֳɶ��ٸ��ӿ�
	unsigned subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;                                         // �����ϲ�������������ӿ�����

	// �ҵ�Ҫ����Ķ˵�ֵ
	SinglesStruct sampleValue = data[blockIdx.x * (blockDim.x * subBlockSize) + threadIdx.x * subBlockSize];     // ע��һ���̲߳���һ���˵�
	unsigned rankSampleCurrent = blockIdx.x * blockDim.x + threadIdx.x;                                     // �˵��
	unsigned rankSampleOpposite;

	// �ҵ��ö˵�ţ��ӿ飩���ڵ������Լ��������Ӧ���� 
	unsigned indexBlockCurrent = rankSampleCurrent / subBlocksPerSortedBlock;                               // �˵�Ŷ�Ӧ������
	unsigned indexBlockOpposite = indexBlockCurrent ^ 1;                                                    // ���ӦҪ���������

	// �ҵ�����˵�ֵ�����Ӧ���е��ĸ��ӿ�
	if (indexBlockCurrent % 2 == 0)
	{
		rankSampleOpposite = binarySearchInclusive<subBlockSize>(                              // �в����Ķ��ֲ���
			data, sampleValue, indexBlockOpposite * sortedBlockSize,                                      // ���Ҷ˵�����һ�������е�λ��
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite = (rankSampleOpposite - sortedBlockSize) / subBlockSize;                       // ���λ��
	}
	else
	{
		rankSampleOpposite = binarySearchExclusive<subBlockSize>(
			data, sampleValue, indexBlockOpposite * sortedBlockSize,
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite /= subBlockSize;
	}

	// ����ϲ����ڵ��������� 
	unsigned sortedIndex = rankSampleCurrent % subBlocksPerSortedBlock + rankSampleOpposite;         // ���ڿ���

	// ���������ڵ�ǰ�����������е�λ��
	unsigned rankDataCurrent = (rankSampleCurrent * subBlockSize % sortedBlockSize) + 1;             // �ڵ�ǰ�����е����λ��+1
	unsigned rankDataOpposite;

	// �������������ڵ��ӿ�������������С�˲��ҷ�Χ��
	unsigned indexSubBlockOpposite = sortedIndex % subBlocksPerMergedBlock - rankSampleCurrent % subBlocksPerSortedBlock - 1;
	unsigned indexStart = indexBlockOpposite * sortedBlockSize + indexSubBlockOpposite * subBlockSize + 1;     // ��ȡ�˵�ֵ
	unsigned indexEnd = indexStart + subBlockSize - 2;

	if ((int)(indexStart - indexBlockOpposite * sortedBlockSize) >= 0)
	{
		if (indexBlockOpposite % 2 == 0)   // ������������
		{
			rankDataOpposite = binarySearchExclusive(           // ���ֲ���λ��
				data, sampleValue, indexStart, indexEnd               // ���������ĺ���һ��λ�ã�������֤�����������д���������ͬ������ʱҲ�������㣩
			);
		}
		else                              // ������ż����
		{
			rankDataOpposite = binarySearchInclusive(
				data, sampleValue, indexStart, indexEnd        // ����������λ��
			);
		}

		rankDataOpposite -= indexBlockOpposite * sortedBlockSize;
	}
	else
	{
		rankDataOpposite = 0;
	}

	// ��Ӧ��ÿ���ӿ����λ��
	if ((rankSampleCurrent / subBlocksPerSortedBlock) % 2 == 0)    // ������ż����
	{
		ranksEven[sortedIndex] = rankDataCurrent;              // ��������1����֤������һ��Ԫ�ز���ϲ�
		ranksOdd[sortedIndex] = rankDataOpposite;
	}
	else
	{
		ranksEven[sortedIndex] = rankDataOpposite;
		ranksOdd[sortedIndex] = rankDataCurrent;               // ��������1
	}
}

//�ϲ���rank���������������ż���������ӿ顣
template <unsigned subBlockSize>
__global__ void mergeKernel(
	SinglesStruct* data, SinglesStruct* dataBuffer, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize
)
{
	__shared__ SinglesStruct tileEven[subBlockSize];           // ÿ���ӿ��Ԫ�ظ�������512
	__shared__ SinglesStruct tileOdd[subBlockSize];            // �ӿ������ӿ��ż�ӿ��Ԫ�ظ�����������256

	unsigned indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;               // �ҵ���ǰ�߳̿��Ӧ���ӿ�
	unsigned indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;                  // ���߳̿�������ӿ����ڵ����еĲ�������ʼλ��

	// �������ڵ�ż���������飬���ǽ����ϲ�  
	unsigned indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
	unsigned offsetEven, offsetOdd;
	unsigned numElementsEven, numElementsOdd;

	// ��ȡż���������ӿ��START����  
	// ÿ���߳̿������ͬ����ʼ�ͽ���λ��
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
	// ��ȡż���������ӿ��END����  
	if (blockIdx.x < gridDim.x - 1)
	{
		indexEndEven = ranksEven[indexRank];
		indexEndOdd = ranksOdd[indexRank];
	}
	else                           // ���ж�Ӧ�����һ���߳̿�
	{
		indexEndEven = sortedBlockSize;   // ���һ������
		indexEndOdd = sortedBlockSize;
	}

	numElementsEven = indexEndEven - indexStartEven;    // ������߳̿������Ԫ�ظ���
	numElementsOdd = indexEndOdd - indexStartOdd;

	// ��ż�������ӿ��ж�ȡ����
	if (threadIdx.x < numElementsEven)
	{
		offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
		tileEven[threadIdx.x] = data[offsetEven];
	}
	// �����������ӿ��ж�ȡ����
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
		// �����ȫ�������е�λ��
		dataBuffer[offsetEven + rankOdd] = tileEven[threadIdx.x];
	}

	if (threadIdx.x < numElementsOdd)
	{
		unsigned rankEven = binarySearchExclusive(tileEven, tileOdd[threadIdx.x], 0, numElementsEven - 1);
		rankEven += indexStartEven;
		dataBuffer[offsetOdd + rankEven] = tileOdd[threadIdx.x];
	}
}
