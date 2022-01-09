#include <iostream>

#include "Samples2Singles.h"
#include "CoinPetPara.h"

int main() {
    CoinPetPara para;
    para.m_nThreadNum = 0;
    para.m_bOnlineCoin = true;
    para.m_strPETSavePath = "../CoinTest";
    para.m_strCoinSavePath = "../CoinTest";
    para.m_strCoinPositionPath = "../profile/D80/PositionMap";
    para.m_strCoinEnergyCaliPath = "../profile/D80/EnergyCalibration";
    
    para.m_nPositionSize = 256;
    para.m_nCrystalSize = 13;
    para.m_nChannelNum = 12;
    para.m_nDUNum = 4;
    para.m_nTimeWindow = 5000;
    para.m_fMinEnergy = 350.0f;
    para.m_fMaxEnergy = 650.0f;
    
    para.m_nCrystalNumX = 1;
    para.m_nCrystalNumY = 13;
    para.m_nCrystalNumZ = 13;
    
    para.m_nBlockNumX = 1;
    para.m_nBlockNumY = 1;
    para.m_nBlockNumZ = 4;
    
    para.m_nModuleNumX = 1;
    para.m_nModuleNumY = 1;
    para.m_nModuleNumZ = 1;
    
    Samples2Singles s2s(para, true);
    s2s.start();
    return 0;
}
