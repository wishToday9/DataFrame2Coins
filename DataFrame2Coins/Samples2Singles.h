//
// Created by svf on 2021/8/24.
//

#ifndef SAMPLES2SINGLES_SAMPLES2SINGLES_H
#define SAMPLES2SINGLES_SAMPLES2SINGLES_H

#include "UAcquireDef.h"
#include "CoinPetPara.h"
#include "UPositionMap.h"
#include "UEnergyProfile.h"
#include "contants.h"
#include "kernels.h"
#include "ConvertUdpToSingles.h"
#include "until.h"
#include "TimerClock.h"
#include <fstream>
#include <vector>
#include <thread>
#include <cmath>
#include <iostream>
#include <Windows.h>
#include <cuda_device_runtime_api.h>

class Samples2Singles {
public:
	Samples2Singles(CoinPetPara&, bool);
	~Samples2Singles();

	void start();

private:
	std::vector<CoinStruct> myCoins;
	unsigned coinNum = 0;
	bool coinEnergy(double);
	bool convertUdpToSingles(DataFrameV2& src, SinglesStruct& dst);
	bool mIsCoinEnergy;
	void checkCudaError(cudaError_t error)
	{
		if (error != cudaSuccess)
		{
			//std::cout << i << std::endl;
			printf("Error in CUDA function.\nError: %s\n", cudaGetErrorString(error));
			getchar();
			exit(EXIT_FAILURE);
		}
	}

	void runCoinTimeKernel(SinglesStruct* d_mSortedSingles, int* nextIndex, unsigned mSortedSinglesNum, CoinStruct* d_data, unsigned* newLength) {
		unsigned threadsNum = THREAD_NUM;
		unsigned threadsElems = THREAD_ELEM;
		unsigned elemsPerBlock = threadsNum * threadsElems;
		int shareMemory = THREAD_NUM * THREAD_ELEM * sizeof(int);
		unsigned* d_newLength;
		cudaError_t error;
		error = cudaMalloc((void**)&d_newLength, sizeof(unsigned));
		
		dim3 dimGrid((mSortedSinglesNum - 1) / elemsPerBlock + 1, 1, 1);
		dim3 dimBlock(threadsNum, 1, 1);

		CoinTimeKernel << <dimGrid, dimBlock >> > (d_mSortedSingles, nextIndex, mSortedSinglesNum);

		SelectKernel << <dimGrid, dimBlock >> > (d_mSortedSingles,nextIndex, mSortedSinglesNum);

		int* h_index = new int[mSortedSinglesNum];
		cudaMemcpy(h_index, nextIndex, mSortedSinglesNum * sizeof(int), cudaMemcpyDeviceToHost);

		unsigned* scanId;
		unsigned* globalLen;
		cudaMalloc((void**)&scanId, ((mSortedSinglesNum - 1) / elemsPerBlock + 1) * threadsNum * sizeof(unsigned));
		cudaMalloc((void**)&globalLen, ((mSortedSinglesNum - 1) / elemsPerBlock + 1) * sizeof(unsigned));

		FillDataKernel << <dimGrid, dimBlock >> > (d_mSortedSingles, d_data, nextIndex, mSortedSinglesNum, d_newLength, scanId, globalLen);

	
		CorrectPosKernel << < dimGrid, dimBlock >> > (d_mSortedSingles, d_data, d_newLength, scanId, globalLen, mSortedSinglesNum, nextIndex);
		cudaMemcpy(newLength, d_newLength, sizeof(unsigned), cudaMemcpyDeviceToHost);
		//std::cout << "newLength:" << *newLength << std::endl;

		//unsigned temp = 0;
		//for (int i = 0; temp < *newLength; ++i) {
		//	if (h_index[i] != -1) {
		//		temp++;
		//	}
		//	if (temp > *newLength - 10) {
		//		std::cout << h_index[i] << "  ";
		//	}
		//}
		error = cudaFree(d_newLength);
		checkCudaError(error);
		error = cudaFree(globalLen);
		checkCudaError(error);
		error = cudaFree(scanId);
		checkCudaError(error);

	}

	void runConvertUdpToSinglesAndSortKernel(DataFrameV2* src, SinglesStruct*& dst, unsigned arrayLength, unsigned* coinEnergyLength, unsigned& errors, uint8* d_m_pPositionTable, float* d_m_pEnergyCorrFactor) {
		SinglesStruct* dstBuffer = NULL;
		//SinglesStruct* Buffer = NULL;
		unsigned* d_actualSinglesLength = NULL;
		//TempSinglesStruct* d_temp;
		//TempSinglesStruct* temp = new TempSinglesStruct[arrayLength];
		//Buffer = new SinglesStruct[arrayLength];

		cudaError_t error;
		error = cudaMalloc((void**)&dstBuffer, nextPowerOf2(arrayLength) * sizeof(*dstBuffer));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_actualSinglesLength, sizeof(unsigned));
		checkCudaError(error);
		//error = cudaMalloc((void**)&d_temp, arrayLength * sizeof(TempSinglesStruct));
		//checkCudaError(error);

		unsigned threadsConvert = THREADS_CONVERT;
		unsigned elemsConvert = ELEMS_CONVERT;
		unsigned elemsPerThreadBlock = threadsConvert * elemsConvert;


		dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);       // 启动的线程块数目
		dim3 dimBlock(threadsConvert, 1, 1);                                   // 每个线程块启动的线程数

		// step1：完成信号转换和能量筛选
		// 还没有将dstBuffer传入
		TimerClock tc0;
		tc0.update();

		if (mIsCoinEnergy) {

			convertUdpToSinglesKernel<THREADS_CONVERT, ELEMS_CONVERT> << <dimGrid, dimBlock >> > (src, dst, dstBuffer, arrayLength, mCoinPetPara, d_m_pPositionTable,
				d_m_pEnergyCorrFactor, mIsCoinEnergy, d_actualSinglesLength);


			error = cudaMemcpy(coinEnergyLength, (void*)d_actualSinglesLength, sizeof(unsigned), cudaMemcpyDeviceToHost);
			checkCudaError(error);
		}
		else {

			convertUdpToSinglesKernel<THREADS_CONVERT, ELEMS_CONVERT> << <dimGrid, dimBlock >> > (src, dst, arrayLength, mCoinPetPara, d_m_pPositionTable,
				d_m_pEnergyCorrFactor, mIsCoinEnergy, d_actualSinglesLength);

			*coinEnergyLength = arrayLength;
		}
		std::cout << endl;
		std::cout << "Convert time:" << tc0.getSecond() << "s" << std::endl;

		TimerClock tc1;
		tc1.update();
		// step2：完成排序，排序完成后数据有可能在dst中也有可能在dstBuffer中
		parallelMergeSort(dst, dstBuffer, *coinEnergyLength);
		std::cout << endl;
		std::cout << "Sort time:" << tc1.getSecond() << "s" << std::endl;

		// 判断是否需要转换指针
		unsigned elemsPerInitMergeSort = THREADS_MERGESORT * ELEMS_MERGESORT;
		unsigned arrayLenRoundedUp = max(nextPowerOf2(*coinEnergyLength), elemsPerInitMergeSort);
		unsigned numMergePhases = log2((double)arrayLenRoundedUp) - log2((double)elemsPerInitMergeSort);

		if (numMergePhases % 2 == 1)
		{
			SinglesStruct* temp = dst;
			dst = dstBuffer;
			dstBuffer = temp;
		}

		error = cudaFree(dstBuffer);
		checkCudaError(error);
		error = cudaFree(d_actualSinglesLength);
		checkCudaError(error);
	}


	void parallelMergeSort(SinglesStruct* dst, SinglesStruct* dstBuffer, unsigned arrayLength) {
		unsigned* d_ranksEven = NULL;
		unsigned* d_ranksOdd = NULL;

		unsigned elemsPerThreadBlock = THREADS_MERGESORT * ELEMS_MERGESORT;
		unsigned arrayLenRoundedUp = max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
		unsigned ranksLength = (arrayLenRoundedUp - 1) / SUB_BLOCK_SIZE + 1;         // 求出子块的数目

		cudaError_t error;
		error = cudaMalloc((void**)&d_ranksEven, 2 * ranksLength * sizeof(*d_ranksEven));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_ranksOdd, 2 * ranksLength * sizeof(*d_ranksOdd));
		checkCudaError(error);

		unsigned sortedBlockSize;
		sortedBlockSize = THREADS_MERGESORT * ELEMS_MERGESORT;

		unsigned lastPaddingMergePhase = log2((double)(sortedBlockSize));                       // 计算合并阶段
		unsigned arrayLenPrevPower2 = previousPowerOf2(arrayLength);

		addPadding(dst, dstBuffer, arrayLength);           // 两个数组都填满2的幂次（可能会导致越界）

		//SinglesStruct* h_lSingles = new SinglesStruct[arrayLength];

		//cudaMemcpy(h_lSingles, (void*)dst, arrayLength * sizeof(*h_lSingles), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < 4086; i += 100) {
		//	std::cout << h_lSingles[i].timevalue << " ";
		//}
		//std::cout << std::endl;

		//delete[] h_lSingles;

		runMergeSortKernel(dst, arrayLength);         // data局部有序，databuffer无序 每1024个元素有序


		while (sortedBlockSize < arrayLength)
		{
			SinglesStruct* temp = dst;
			dst = dstBuffer;
			dstBuffer = temp;


			//lastPaddingMergePhase = copyPaddedElements(                            // 判断剩余部分是否需要参与排序，如果需要排序，是否应该进行数据转移，并记录最近一次剩余部分参与排序的阶段
			//	dst + arrayLenPrevPower2, dstBuffer + arrayLenPrevPower2,
			//	arrayLength, sortedBlockSize, lastPaddingMergePhase
			//);

			runGenerateRanksKernel(dstBuffer, d_ranksEven, d_ranksOdd, arrayLength, sortedBlockSize);

			runMergeKernel(dstBuffer, dst, d_ranksEven, d_ranksOdd, arrayLength, sortedBlockSize);

			sortedBlockSize *= 2;
			//std::cout << sortedBlockSize << std::endl;
			//SinglesStruct* Buffer;
			//Buffer = new SinglesStruct[nextPowerOf2(arrayLength)];
			//error = cudaMemcpy(Buffer, (void*)dst, nextPowerOf2(arrayLength) * sizeof(SinglesStruct), cudaMemcpyDeviceToHost);
			//checkCudaError(error);
			//for (int i = 0; i < sortedBlockSize ; i += sortedBlockSize / 5) {
			//	std::cout << Buffer[i].timevalue << "  ";
			//}
			//std::cout << endl;
			//delete[]Buffer;

		}



		error = cudaFree(d_ranksEven);
		checkCudaError(error);

		error = cudaFree(d_ranksOdd);
		checkCudaError(error);

	}

	void runGenerateRanksKernel(
		SinglesStruct* d_data, unsigned* d_ranksEven, unsigned* d_ranksOdd, unsigned arrayLength, unsigned sortedBlockSize
	)
	{
		unsigned subBlockSize = SUB_BLOCK_SIZE;
		unsigned threadsKernel = THREADS_GEN_RANKS;

		unsigned arrayLenRoundedUp = calculateMergeArraySize(arrayLength, sortedBlockSize);     // 计算需要进行合并的数据规模
		// 根据合并规模计算出所需的子块数，所需的子块数即为端点个数，后面每个端点分配一个线程，分别算出每个端点对应划分出来的子块编号，
		// 每个端点在本身序列中的位置和相对应序列中的位置
		unsigned numAllSamples = (arrayLenRoundedUp - 1) / subBlockSize + 1;
		unsigned threadBlockSize = min(numAllSamples, threadsKernel);           // 计算出所需的线程数（每个线程块只启动128个线程）

		dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);        // 根据所需要处理的端点数的多少划分block
		dim3 dimBlock(threadBlockSize, 1, 1);

		generateRanksKernel<SUB_BLOCK_SIZE> << <dimGrid, dimBlock >> > (d_data, d_ranksEven, d_ranksOdd, sortedBlockSize);

	}

	// 计算合并数组规模的函数
	unsigned calculateMergeArraySize(unsigned arrayLength, unsigned sortedBlockSize)
	{
		unsigned arrayLenMerge = previousPowerOf2(arrayLength);
		unsigned mergedBlockSize = 2 * sortedBlockSize;

		// 数组长度本身是否是2的幂次，如果是则直接返回
		if (arrayLenMerge != arrayLength)
		{
			// 计算剩余部分长度
			unsigned remainder = arrayLength - arrayLenMerge;

			// 如果剩余部分长度大于有序长度，则剩余部分需要参与合并，如果剩余部分长度小于有序长度，则不需合并，等待主要部分合并完成以后再合并剩余部分
			// 判断剩余部分长度是否大于有序长度
			if (remainder >= sortedBlockSize)
			{
				arrayLenMerge += roundUp(remainder, 2 * sortedBlockSize);    // 剩余部分可以执行合并
			}
			// 判断主要部分是否合并完成
			else if (arrayLenMerge == sortedBlockSize)
			{
				arrayLenMerge += sortedBlockSize;
			}
		}

		// 返回最终的合并规模
		return arrayLenMerge;
	}

	void runMergeKernel(
		SinglesStruct* d_data, SinglesStruct* d_dataBuffer, unsigned* d_ranksEven,
		unsigned* d_ranksOdd, unsigned arrayLength, unsigned sortedBlockSize
	)
	{
		unsigned arrayLenMerge = calculateMergeArraySize(arrayLength, sortedBlockSize);         // 判断剩余部分是否需要加入合并，得到合并的规模
		unsigned mergedBlockSize = 2 * sortedBlockSize;
		// 合并完成后的序列数
		unsigned numMergedBlocks = (arrayLenMerge - 1) / mergedBlockSize + 1;
		unsigned subBlockSize = SUB_BLOCK_SIZE;
		// 合并一对序列需要的线程块数
		unsigned subBlocksPerMergedBlock = (mergedBlockSize - 1) / subBlockSize + 1;

		// 通过端点进行划分，n个端点划分出n+1个子块，n+1个子块使用n+1个线程块进行合并
		dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
		dim3 dimBlock(subBlockSize, 1, 1);

		mergeKernel<SUB_BLOCK_SIZE> << <dimGrid, dimBlock >> > (d_data, d_dataBuffer, d_ranksEven, d_ranksOdd, sortedBlockSize);

	}

	void addPadding(SinglesStruct* dst, SinglesStruct* dstBuffer, unsigned arrayLength)
	{
		unsigned threadsMergeSort = THREADS_MERGESORT;
		unsigned elemsMergeSort = ELEMS_MERGESORT;
		unsigned elemsPerThreadBlock = threadsMergeSort * elemsMergeSort;
		unsigned arrayLenRoundedUp = max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
		runAddPaddingKernel(dst, dstBuffer, arrayLength, arrayLenRoundedUp);
	}


	void runAddPaddingKernel(SinglesStruct* d_arrayPrimary, SinglesStruct* d_arrayBuffer, unsigned indexStart, unsigned indexEnd)
	{
		if (indexStart == indexEnd)
		{
			return;
		}

		unsigned paddingLength = indexEnd - indexStart;
		unsigned elemsPerThreadBlock = THREADS_PADDING * ELEMS_PADDING;
		dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
		dim3 dimBlock(THREADS_PADDING, 1, 1);

		SinglesStruct maxVal;

		maxVal.timevalue = MAX_VAL;
		addPaddingKernel << <dimGrid, dimBlock >> > (d_arrayPrimary, d_arrayBuffer, indexStart, paddingLength, maxVal);   // 这里会有访问越界
	}

	void runMergeSortKernel(SinglesStruct* d_data, unsigned arrayLength)
	{
		unsigned elemsPerThreadBlock, sharedMemSize;

		elemsPerThreadBlock = THREADS_MERGESORT * ELEMS_MERGESORT;              // 每个线程块操作的元素个数
		// "2 *" because buffer shared memory is used in kernel alongside primary shared memory
		sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_data);                 // 创建两倍元素数目的共享内存大小

		// 判断需要执行合并操作的数组长度，是线程块操作元素个数的整数倍，并不需要是2的幂次，因为后面补齐的数据都是有序的
		unsigned arrayLenRoundedUp = roundUp(arrayLength, elemsPerThreadBlock);
		dim3 dimGrid((arrayLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
		dim3 dimBlock(THREADS_MERGESORT, 1, 1);

		mergeSortKernel<THREADS_MERGESORT, ELEMS_MERGESORT> << <dimGrid, dimBlock, sharedMemSize >> > (d_data);
	}

	unsigned copyPaddedElements(
		SinglesStruct* d_dataFrom, SinglesStruct* d_dataTo, unsigned arrayLength,
		unsigned sortedBlockSize, unsigned lastPaddingMergePhase
	)
	{
		// 计算主要部分长度和剩余部分长度
		unsigned arrayLenMerge = previousPowerOf2(arrayLength);
		unsigned remainder = arrayLength - arrayLenMerge;

		// 剩余部分需要参与排序的两种情况，情况一：剩余部分长度＞=当前有序序列长度、情况二：主要部分已经全部有序
		// 由于剩余部分如果要参与排序就必须将剩余部分移动到参与排序的那个数组中
		if (remainder >= sortedBlockSize || arrayLenMerge == sortedBlockSize)
		{
			// 计算当前合并的阶段与上次剩余部分参与排序的阶段做差进行判断
			unsigned currentMergePhase = log2((double)(2 * sortedBlockSize));
			unsigned phaseDifference = currentMergePhase - lastPaddingMergePhase;

			if (phaseDifference % 2 == 0)
			{
				cudaError_t error;


				// 这里用了零拷贝之后可能会出错
				error = cudaMemcpy(d_dataTo, d_dataFrom, remainder * sizeof(*d_dataTo), cudaMemcpyDeviceToDevice);
				checkCudaError(error);

				//	h_dataBuffer[i] = _h_data[i];
				//}
			}

			// 更新剩余部分参与合并排序的阶段
			lastPaddingMergePhase = currentMergePhase;
		}

		return lastPaddingMergePhase;
	}

	CoinPetPara mCoinPetPara;

	UPositionMap* mPositionMap;
	UEnergyProfile* mEnergyProfile;

	TimerClock mTimerClock;
};
//
// Created by svf on 2021/8/24.
//

Samples2Singles::Samples2Singles(CoinPetPara& cpp, bool isCoinEnergy) {
	mCoinPetPara = cpp;
	mIsCoinEnergy = isCoinEnergy;
	mPositionMap = new UPositionMap(cpp.m_nChannelNum, cpp.m_nDUNum, cpp.m_nCrystalSize, cpp.m_nPositionSize);
	mEnergyProfile = new UEnergyProfile(cpp.m_nChannelNum, cpp.m_nDUNum, cpp.m_nCrystalSize, cpp.m_nPositionSize);
	mPositionMap->ReadPositionTable(cpp.m_strCoinPositionPath + "/PositionTable.dat");
	mEnergyProfile->ReadEnergyCorrFactor(cpp.m_strCoinEnergyCaliPath + "/EnergyCorrFactor.dat");
}

Samples2Singles::~Samples2Singles() {
	delete mPositionMap;
	delete mEnergyProfile;
}

void Samples2Singles::start() {
	// unsigned lChannelNum{mCoinPetPara.m_nChannelNum};
	const unsigned lChannelNum = 12;
	const unsigned lthreadNum = 24;
	unsigned lFileSize[lChannelNum];
	unsigned lSinglesNum[lChannelNum];
	unsigned lTotalSinglesNum{ 0 };
	unsigned lSingleReadBegin[lChannelNum] = { 0 };

	std::ifstream lSamplesFiles[lChannelNum];
	std::ofstream lSinglesFiles[lChannelNum];

	SinglesStruct* lSingles;

	// 1. Get the size of each raw data file and the number of total singles.
	for (unsigned i = 0; i < lChannelNum; ++i) {
		// 读取样本文件
		std::string lStrSampleFilePath = mCoinPetPara.m_strPETSavePath +
			"/thread" + std::to_string(mCoinPetPara.m_nThreadNum) +
			"-channel" + std::to_string(i);
		lSamplesFiles[i].open(lStrSampleFilePath, std::ios::binary);
		if (!lSamplesFiles[i].is_open()) {
			printf("Open file %s failed\n", lStrSampleFilePath.c_str());
			return;
		}
		lSamplesFiles[i].seekg(0, std::ios_base::end);
		lFileSize[i] = lSamplesFiles[i].tellg();
		//printf("FileSize[%d] = %f\n", i, double(lFileSize[i]));

				// 每个样本文件的数据量
		lSinglesNum[i] = lFileSize[i] / sizeof(DataFrameV2);
		printf("FileNum[%d] = %f\n", i, double(lSinglesNum[i]));
		lTotalSinglesNum += lSinglesNum[i];
		if (i >= 1) {
			lSingleReadBegin[i] += lSingleReadBegin[i - 1] + lSinglesNum[i - 1];
		}
	}

	// 总的数据量
	printf("Total Singles Number = %f\n", double(lTotalSinglesNum));
	// 所需的内存大小
	printf("%f MB memory needed\n", double(lTotalSinglesNum) * sizeof(DataFrameV2) / 1024 / 1024);
	mTimerClock.update();
	DataFrameV2* d_lUdpFrame = NULL;
	SinglesStruct* d_lSingles = NULL;

	uint8* d_m_pPositionTable = NULL;
	float* d_m_pEnergyCorrFactor = NULL;
	cudaError_t error;

	error = cudaMalloc((void**)&d_m_pPositionTable, mCoinPetPara.m_nChannelNum * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nPositionSize * mCoinPetPara.m_nPositionSize * sizeof(*d_m_pPositionTable));
	checkCudaError(error);
	error = cudaMalloc((void**)&d_m_pEnergyCorrFactor, sizeof(float) * mCoinPetPara.m_nChannelNum * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nCrystalSize * mCoinPetPara.m_nCrystalSize * 1000);
	checkCudaError(error);
	error = cudaMemcpy((void*)d_m_pPositionTable, mPositionMap->GetPositionTable(0, 0), mCoinPetPara.m_nChannelNum * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nPositionSize * mCoinPetPara.m_nPositionSize * sizeof(*d_m_pPositionTable), cudaMemcpyHostToDevice);
	checkCudaError(error);
	error = cudaMemcpy((void*)d_m_pEnergyCorrFactor, mEnergyProfile->GetEnergyCorrFactor(0, 0, 0), sizeof(float) * mCoinPetPara.m_nChannelNum * mCoinPetPara.m_nDUNum * mCoinPetPara.m_nCrystalSize * mCoinPetPara.m_nCrystalSize * 1000, cudaMemcpyHostToDevice);
	checkCudaError(error);
	TimerClock tc0;
	tc0.update();
	DataFrameV2* lUdpFrame;
	lUdpFrame = new DataFrameV2[lTotalSinglesNum];
	//cudaMallocHost(&lUdpFrame, lTotalSinglesNum * sizeof(DataFrameV2));
	std::cout << "host Alloc time cost: " << tc0.getSecond() << "s" << std::endl;

	// 2. To read raw data files from l_pUdpFrame
	TimerClock tc1;
	tc1.update();
	std::vector<std::thread> lReadTasks;
	for (unsigned i = 0; i < lChannelNum; ++i) {
		lReadTasks.emplace_back([&, i] {
			//auto* lUdpFrame = new DataFrameV2[lSinglesNum[i]];
			//DataFrameV2* lUdpFrame;
		   // cudaHostAlloc(&lUdpFrame, lSinglesNum[i] * sizeof(DataFrameV2), cudaHostAllocWriteCombined | cudaHostAllocMapped);

			//lSingles[i] = new SinglesStruct[lSinglesNum[i]];
			//cudaHostAlloc(&lSingles[i], lSinglesNum[i] * sizeof(DataFrameV2), cudaHostAllocWriteCombined | cudaHostAllocMapped);
			lSamplesFiles[i].seekg(0);
			lSamplesFiles[i].read((char*)(lUdpFrame + lSingleReadBegin[i]), lFileSize[i]);
			lSamplesFiles[i].close(); });
	}
	for (auto& item : lReadTasks) {
		item.join();
	}
	std::cout << std::endl;
	std::cout << "read time cost: " << tc1.getSecond() << "s" << std::endl;
	std::cout << "read rate: " << lTotalSinglesNum * 16.0 / tc1.getSecond() / 1000000.0 << "MB/s" << std::endl;

	unsigned errors = 0;
	unsigned* actualSinglesLength = new unsigned;
	unsigned* newLength = new unsigned;
	TimerClock tc2;
	tc2.update();

	error = cudaMalloc((void**)&d_lUdpFrame, lTotalSinglesNum * sizeof(*lUdpFrame));
	checkCudaError(error);
	error = cudaMalloc((void**)&d_lSingles, nextPowerOf2(lTotalSinglesNum) * sizeof(*d_lSingles));
	checkCudaError(error);

	TimerClock tc5;
	tc5.update();
	error = cudaMemcpy((void*)d_lUdpFrame, lUdpFrame, lTotalSinglesNum * sizeof(*lUdpFrame), cudaMemcpyHostToDevice);
	checkCudaError(error);
	std::cout << std::endl;
	std::cout << "host to device transfer time cost: " << tc5.getSecond() << "s" << std::endl;
	std::cout << "host to device transfer rate : " << lTotalSinglesNum * 16 / tc5.getSecond() / 1000000 << "MB/s" << std::endl;

	TimerClock tc4;
	tc4.update();

	runConvertUdpToSinglesAndSortKernel(d_lUdpFrame, d_lSingles, lTotalSinglesNum, actualSinglesLength, errors, d_m_pPositionTable, d_m_pEnergyCorrFactor);

	TimerClock mTimerClock1;
	mTimerClock1.update();
	// Coin State, since coinning energy finished at Samples2Singles state, it's no need to coin energy here. 
	int* nextIndex;
	CoinStruct* d_data;
	cudaMalloc((void**)&d_data, *actualSinglesLength / 5.0 * sizeof(CoinStruct));
	cudaMalloc((void**)&nextIndex, *actualSinglesLength * sizeof(int));

	TimerClock tc9;
	tc9.update();
	runCoinTimeKernel(d_lSingles, nextIndex, *actualSinglesLength, d_data, newLength);
	cudaDeviceSynchronize();
	std::cout << std::endl;
	std::cout << "coin time cost:" << tc9.getSecond() << "s" << std::endl;



	TimerClock tc10;
	tc10.update();
	CoinStruct* data = new CoinStruct[*newLength];
	cudaMemcpy(data, d_data, *newLength * sizeof(CoinStruct), cudaMemcpyDeviceToHost);
	std::cout << std::endl;
	std::cout << "device to host transfer time cost: " << tc10.getSecond() << "s" << std::endl;
	std::cout << "device to host transfer rate : " << *newLength * sizeof(CoinStruct) / tc10.getSecond() / 1000000 << "MB/s" << std::endl;

	std::cout << std::endl;
	printf("# Coin Time Cost: %fs\n", mTimerClock1.getSecond());
	printf("# Coin rate: %f%%\n", 200.0 * *newLength / *actualSinglesLength);

	std::cout << std::endl;
	std::cout << "parallel convert and sort time cost: " << tc4.getSecond() << "s" << std::endl;
	std::cout << "parallel convert and sort rate : " << lTotalSinglesNum * 16 / tc4.getSecond() / 1000000 << "MB/s" << std::endl;


	std::cout << std::endl;
	std::cout << "parallel convert and sort time cost (include memory copy): " << tc2.getSecond() << "s" << std::endl;
	std::cout << "parallel convert and sort rate (include memory copy): " << lTotalSinglesNum * 16 / tc2.getSecond() / 1000000 << "MB/s" << std::endl;

	//for (int i = 0; i < *newLength; ++i) {
	//	std::cout << data[i].nCoinStruct[0].timevalue << "   " << data[i].nCoinStruct[1].timevalue << std::endl;
	//}
	//TimerClock tc7;
	//tc7.update();
	//for (unsigned i = 0; i < *actualSinglesLength - 1; i++) {
	//	//std::cout << lSingles[i].timevalue << " ";
	//	if (lSingles[i].timevalue > lSingles[i + 1].timevalue) {
	//		++errors;
	//	}
	//}
	//std::cout << std::endl;
	//std::cout << "errors:" << errors << std::endl;
	//std::cout << "check error time:" << tc7.getSecond() << "s" << std::endl;



	//for (int i = 0; i < *newLength; ++i) {
	//	std::cout << data[i].nCoinStruct[0].timevalue << "   " << data[i].nCoinStruct[1].timevalue << std::endl;
	//}

	TimerClock tc3;
	tc3.update();
	std::string lCoinPath = mCoinPetPara.m_strPETSavePath + "/thread" + std::to_string(mCoinPetPara.m_nThreadNum) + ".coins";
	std::ofstream lCoinFile(lCoinPath, std::ios::binary);
	if (!lCoinFile.is_open()) {
		printf("Open file %s failed\n", lCoinPath.c_str());
		return;
	}
	lCoinFile.write((char*)data, *newLength * sizeof(CoinStruct));
	std::cout << std::endl;
	std::cout << " write time cost : " << tc3.getSecond() << "s" << std::endl;
	std::cout << " write rate : " << *newLength * sizeof(CoinStruct) / tc3.getSecond() / 1000000 << "MB/s" << std::endl;

	std::cout << std::endl;
	printf("# total time cost: %fs\n", mTimerClock.getSecond());
	printf("# total rate: %fMB/s\n", lTotalSinglesNum * 16 / mTimerClock.getSecond() / 1000000);

	delete[] lUdpFrame;
	delete[] data;
	delete newLength;
	delete actualSinglesLength;
	error = cudaFree(d_lUdpFrame);
	checkCudaError(error);
	error = cudaFree(d_lSingles);
	checkCudaError(error);
	error = cudaFree(d_m_pEnergyCorrFactor);
	checkCudaError(error);
	error = cudaFree(d_m_pPositionTable);
	checkCudaError(error);
	error = cudaFree(nextIndex);
	checkCudaError(error);
	error = cudaFree(d_data);
	checkCudaError(error);

}

bool Samples2Singles::coinEnergy(double energy) {
	bool isValid = energy >= mCoinPetPara.m_fMinEnergy && energy <= mCoinPetPara.m_fMaxEnergy;
	return isValid;
}

bool Samples2Singles::convertUdpToSingles(DataFrameV2& src, SinglesStruct& dst) {
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

	TempSinglesStruct temp;

	/* Temporary structure to provide BDM and DU info */
	temp.globalBDMIndex = src.nBDM;
	temp.localDUIndex = src.nHeadAndDU & (0x0F);

	/* Time convertion, from unsigned char[8] to double */
	uint64 nTimeTemp;
	nTimeTemp = src.nTime[0];
	for (unsigned i = 1; i <= 7; ++i) {
		nTimeTemp <<= 8;
		nTimeTemp |= src.nTime[i];
	}
	temp.timevalue = (double)nTimeTemp;

	/* Position correction */
	uint32 originCrystalIndex = mPositionMap->GetPositionTable(temp.globalBDMIndex, temp.localDUIndex)[src.X + src.Y * positionSize];
	uint32 localX = originCrystalIndex % crystalNumZ;
	uint32 localY = originCrystalIndex / crystalNumY;
	temp.localCrystalIndex = localX + (crystalNumY - 1 - localY) * crystalNumZ;

	/* Time correction */

	/* Energy convertion, from unsigned char[2] to float */
	uint32 nEnergyTemp;
	nEnergyTemp = (src.Energy[0] << 8 | src.Energy[1]);
	temp.energy = (float)nEnergyTemp;

	/* Up to different system structure ||Changeable|| */
	//    uint32 nCrystalIdInRing = temp.globalBDMIndex % m_nChannelNum * m_nCrystalSize + (m_nCrystalSize - temp.localCrystalIndex / m_nCrystalSize -1);
	//    uint32 nRingId = temp.localDUIndex % m_nDUNum * m_nCrystalSize + temp.localCrystalIndex % m_nCrystalSize;
	//    uint32 nCrystalNumOneRing = m_nCrystalSize * m_nChannelNum;

	uint32 nCrystalIdInRing = temp.globalBDMIndex % (m_nChannelNum * moduleNumY) * blockNumY * crystalNumY + temp.localDUIndex / blockNumZ * crystalNumY + temp.localCrystalIndex / crystalNumZ;
	uint32 nRingId = temp.globalBDMIndex / (m_nChannelNum * moduleNumY) * blockNumZ * crystalNumZ + temp.localDUIndex % blockNumZ * crystalNumZ + temp.localCrystalIndex % crystalNumZ;
	uint32 nCrystalNumOneRing = crystalNumY * blockNumY * m_nChannelNum;

	dst.globalCrystalIndex = nCrystalIdInRing + nRingId * nCrystalNumOneRing;
	/* Energy correction */
	dst.energy = temp.energy * mEnergyProfile->GetEnergyCorrFactor(temp.globalBDMIndex, temp.localDUIndex, temp.localCrystalIndex)[int(floor(temp.energy / 10))];
	dst.timevalue = temp.timevalue;

	// TODO: THE RIGHT PLACE FOR COIN ENERGY!
	if (mIsCoinEnergy) {
		return coinEnergy(dst.energy);
	}
	else {
		return true;
	}
}




#endif //SAMPLES2SINGLES_SAMPLES2SINGLES_H
