#pragma once
#ifndef COIN_PET_PARA
#define COIN_PET_PARA

#include <string>

class CoinPetPara {
public:
	std::string m_strPETSavePath;
	std::string m_strCoinSavePath;
	std::string m_strCoinPositionPath;
	std::string m_strCoinEnergyCaliPath;
	unsigned char m_nBedNum;
	unsigned char m_nFrameNum;
	unsigned char m_nBedIndex;
	unsigned char m_nFrameIndex;
	bool m_bOnlineCoin;
	void* m_pDataBuffer = nullptr;

	unsigned int m_nPositionSize;
	unsigned int m_nCrystalSize;

	//equals to panel num
	unsigned int m_nChannelNum;

	unsigned int m_nDUNum;
	unsigned int m_nTimeWindow;
	float m_fMinEnergy;
	float m_fMaxEnergy;

	int m_nCrystalNumX;
	int m_nCrystalNumY;
	int m_nCrystalNumZ;
	int m_nBlockNumX;
	int m_nBlockNumY;
	int m_nBlockNumZ;
	int m_nModuleNumX;
	int m_nModuleNumY;
	int m_nModuleNumZ;

	unsigned int m_nThreadNum;
};
#endif // !COIN_PET_PARA
