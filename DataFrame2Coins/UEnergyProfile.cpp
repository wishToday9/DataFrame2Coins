/********************************************************************

  Copyright (C), 2019, All rights reserved

  File Name     :    UEnergyProfile.cpp
  Description   :
  History       :

  <author>            <time>            <desc>
  Lingyiqing          2019/7/22         create

********************************************************************/

#include "UEnergyProfile.h"
#include "UAcquireDef.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <windows.h>
#include <memory.h>
#include <math.h>
using namespace std;
#define DEBUG

/*********************** 
 Constrctor Function 
***********************/
UEnergyProfile::UEnergyProfile(uint32 f_nBDMNum, uint32 f_nDUNum, uint32 f_nCrystalSize, uint32 f_nPositionSize)
{
    this->m_nBDMNum = f_nBDMNum;
    this->m_nDUNum = f_nDUNum;
    this->m_nCrystalSize = f_nCrystalSize;
    this->m_nPositionSize = f_nPositionSize;

    m_pPositionMap = new UPositionMap(m_nBDMNum,m_nDUNum,m_nCrystalSize,m_nPositionSize);

    /* Allocate memory for m_pEnergyProfile and m_pEnergyCorrFactor, and set to zero */
    /* 0 <= E < 10, 10 <= E < 20, ... 9990 <= E < 10000 */
    m_pEnergyProfile = new uint32[m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000];   //4*24*4*13*13*1000=64,896,000
    cudaMallocHost(&m_pEnergyCorrFactor, m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 * sizeof(float));
    //m_pEnergyCorrFactor = new float[m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000]; //4*24*4*13*13*1000=64,896,000
    m_pEnergyRecord = new uint32[m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize];           //4*24*4*13*13
    memset(m_pEnergyProfile, 0, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    memset(m_pEnergyCorrFactor, 0, sizeof(float) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    memset(m_pEnergyRecord, 0, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize);
}


/***********************
 Destructor Function 
***********************/
UEnergyProfile::~UEnergyProfile()
{
    RELEASE_POINTER(m_pPositionMap);
    RELEASE_POINTER(m_pEnergyProfile);
    cudaFreeHost(m_pEnergyCorrFactor);
    //RELEASE_POINTER(m_pEnergyCorrFactor);
    RELEASE_POINTER(m_pEnergyRecord);
}

void UEnergyProfile::Clear()
{
    memset(m_pEnergyProfile, 0, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    memset(m_pEnergyCorrFactor, 0, sizeof(float) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    memset(m_pEnergyRecord, 0, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize);
}

/**********************************************************
Function：         CreateEnergyProfile(string f_strReadPath)
Description：      To create an energy profile for each crystal, 
                   saved in m_pEnergyProfile, which records the 
                   counts in different energy intervals.
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UEnergyProfile::CreateEnergyProfile(string f_strReadPath, string f_strPositionTablePath)
{
    m_pPositionMap->ReadPositionTable(f_strPositionTablePath);
    ifstream fp;
    uint32 l_nFileSize = 0;
    uint32 l_nFrameNum = 0;
    /* Read raw data from files. Counts in each energy interval are accumulating */
    for(uint32 i = 0; i < m_nBDMNum; i++)
    {
        fp.open((f_strReadPath + "/thread0-channel" + to_string(i)).c_str());
        if(fp.is_open())
        {
            #ifdef DEBUG
                printf("Open file %s succeeds\n", (f_strReadPath + "/thread0-channel" + to_string(i)).c_str());
            #endif
        }
        else{
            #ifdef DEBUG
                printf("Open file %s fails\n", (f_strReadPath + "/thread0-channel" + to_string(i)).c_str());
            #endif
        }
        /* Calculate the size of the file */
        fp.seekg(0, ios_base::end);
        l_nFileSize = fp.tellg();
        l_nFrameNum = l_nFileSize / sizeof(DataFrameV2);
        /* Allocate a suitable size of memory */
        DataFrameV2* l_pTempFrame = new DataFrameV2[l_nFrameNum];
        /* Read file, save to l_pTempFrame */
        fp.seekg(0);
        fp.read((char*)l_pTempFrame, l_nFileSize);
        /* Operate on each frame */
        for(uint32 j = 0; j < l_nFrameNum; j++)
        {
            /* Get the info of DU, BDM, X, Y and energy */
            uint32 l_nDUId = l_pTempFrame[j].nHeadAndDU & (0x0F);
            uint32 l_nBDMId = l_pTempFrame[j].nBDM;
            uint32 l_nX = l_pTempFrame[j].X;
            uint32 l_nY = l_pTempFrame[j].Y;
            float l_fEnergy = float((l_pTempFrame[j].Energy[0] << 8) | l_pTempFrame[j].Energy[1]);
            
            /* Counts are accumulating */
            m_pEnergyProfile[l_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 +
                             l_nDUId * m_nCrystalSize * m_nCrystalSize * 1000 +
                             m_pPositionMap->GetPositionTable(l_nBDMId, l_nDUId)[l_nX + l_nY * m_nPositionSize] * 1000 +
                             uint32(floor(l_fEnergy / 10))] ++;
        }
        /* Close the file and release the memory. Start again. */
        fp.close();
        RELEASE_ARRAY_POINTER(l_pTempFrame);
    }   

    CreateEnergyCorrFactor();

    //init the m_pEnergyRecord to maxium counts of energy interval
//    for(uint32 i = 0; i < m_nBDMNum; i++)
//    {
//        for(uint32 j = 0; j < m_nDUNum; j++)
//        {
//            for(uint32 k = 0; k < m_nCrystalSize * m_nCrystalSize; k++)
//            {
//                uint32 l_nMaxCounts = 0;
//                uint32 l_nRecord = 0;
//                uint32* l_pTempProfile=m_pEnergyProfile + i * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000
//                                                        + j * m_nCrystalSize * m_nCrystalSize * 1000
//                                                        + k * 1000;

//                for(uint16 curr = 0; curr < 1000; curr++)
//                {
//                    /* Find the energy with the maximum counts */
//                    if(l_nMaxCounts <= l_pTempProfile[curr])
//                    {
//                        l_nMaxCounts = l_pTempProfile[curr];
//                        l_nRecord = curr;
//                    }
//                }

//                m_pEnergyRecord[i * m_nDUNum * m_nCrystalSize * m_nCrystalSize
//                        + j * m_nCrystalSize * m_nCrystalSize
//                        + k] = l_nRecord;
//            }
//        }
//    }

}


/**********************************************************
Function：         SaveEnergyProfile(string f_strSavePath)
Description：      To Save energy profiles to file, from m_pEnergyProfile
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UEnergyProfile::SaveEnergyProfile(string f_strSavePath)
{
    ofstream fp(f_strSavePath);
    if(fp.is_open())
    {
        #ifdef DEBUG
            printf("Open file %s succeeds\n", f_strSavePath.c_str());
        #endif
    }
    else{
        #ifdef DEBUG
            printf("Open file %s fails\n", f_strSavePath.c_str());
        #endif
    }
    fp.write((char*)m_pEnergyProfile, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    fp.close();

#ifdef DEBUG
    printf("write file %s finished\n", f_strSavePath.c_str());
#endif
}


/**********************************************************
Function：         CreateEnergyCorrFactor()
Description：      To create an energy correction factor array
                   for each crystal, saved in m_pEnergyCorrFactor,
                   which record factors between 0 and 1
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UEnergyProfile::CreateEnergyCorrFactor()
{
    /* Allocate a temporary profile and factor array for each crystal */
    uint32* l_pTempProfile = new uint32[1000];
    float* l_pTempCorrFactor = new float[1000];

    for(uint32 i = 0; i < m_nBDMNum; i++)
    {
        for(uint32 j = 0; j < m_nDUNum; j++)
        {
            for(uint32 k = 0; k < m_nCrystalSize * m_nCrystalSize; k++)
            {
                //hpy:we need get m_pEnergyRecord from ui rather than maxium value
                memcpy(l_pTempProfile, m_pEnergyProfile + i * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 + j * m_nCrystalSize * m_nCrystalSize * 1000 + k * 1000, sizeof(uint32) * 1000);
                /*************/
                /* Important */
                NormalizeEnergy(l_pTempProfile, l_pTempCorrFactor);
                /*************/
                memcpy(m_pEnergyCorrFactor + i * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 + j * m_nCrystalSize * m_nCrystalSize * 1000 + k * 1000, l_pTempCorrFactor, sizeof(float) * 1000);

//                float* l_pTempCorrFactor = m_pEnergyCorrFactor + i * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 + j * m_nCrystalSize * m_nCrystalSize * 1000 + k * 1000;
//                uint32 l_nRecord = m_pEnergyRecord[i * m_nDUNum * m_nCrystalSize * m_nCrystalSize + j * m_nCrystalSize * m_nCrystalSize + k];
//                /* Liner normalization. SiPM saturation should be taken into account. To be improved. */
//                for(uint16 j = 0; j < 1000; j++)
//                {
//                    l_pTempCorrFactor[j] = 511.0 / (l_nRecord * 10 + 5);
//                }

            }
        }   
    }
}


/**********************************************************
Function：         NormalizeEnergy(uint32* f_pProfile, float* f_pCorrFactor)
Description：      Get an energy correction factor array 
                   from an energy profile, namely 'Energy Normalization'
Called By：        UEnergyProfile::CreateEnergyCorrFactor()
Return：           None
Others：           None
**********************************************************/
void UEnergyProfile::NormalizeEnergy(uint32* f_pProfile, float* f_pCorrFactor)
{
    uint32* l_pTempProfile = f_pProfile;
    float* l_pTempCorrFactor = f_pCorrFactor;
    uint32 l_nMaxCounts = 0;
    uint32 l_nRecord = 0;
    for(uint16 i = 0; i < 1000; i++)
    {
        /* Find the energy with the maximum counts */
        if(l_nMaxCounts <= l_pTempProfile[i])
        {
            l_nMaxCounts = l_pTempProfile[i];
            l_nRecord = i;
        }
    }
    /* Liner normalization. SiPM saturation should be taken into account. To be improved. */
    for(uint16 j = 0; j < 1000; j++)
    {
        l_pTempCorrFactor[j] = 511.0 / (l_nRecord * 10 + 5);
    }
}

/**********************************************************
Function：         SaveEnergyCorrFactor(string f_strSavePath)
Description：      To Save energy correction factor arrays to file, from m_pEnergyCorrFactor
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UEnergyProfile::SaveEnergyCorrFactor(string f_strSavePath)
{
    ofstream fp(f_strSavePath);
    if(fp.is_open())
    {
        #ifdef DEBUG
            printf("Open file %s succeeds\n", f_strSavePath.c_str());
        #endif
    }
    else{
        #ifdef DEBUG
            printf("Open file %s fails\n", f_strSavePath.c_str());
        #endif
    }
    fp.write((char*)m_pEnergyCorrFactor, sizeof(float) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    fp.close();

#ifdef DEBUG
    printf("write file %s finished\n", f_strSavePath.c_str());
#endif

}

/**********************************************************
Function：         ReadEnergyCorrFactor(string f_strReadPath)
Description：      To Read energy correction factor arrays to m_pEnergyCorrFactor, from file  
Called By：        UCoinPET.cpp
Return：           None
Others：           None
**********************************************************/
void UEnergyProfile::ReadEnergyCorrFactor(string f_strReadPath)
{
    ifstream fp(f_strReadPath, std::ios::binary);
    if(fp.is_open())
    {
        #ifdef DEBUG
            printf("Open file %s succeeds\n", f_strReadPath.c_str());
        #endif
    }
    else{
        #ifdef DEBUG
            printf("Open file %s fails\n", f_strReadPath.c_str());
        #endif
    }
    fp.read((char*)m_pEnergyCorrFactor, sizeof(float) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    fp.close();

}

void UEnergyProfile::ReadEnergyProfile(string f_strReadPath)
{
    ifstream fp(f_strReadPath);
    if(fp.is_open())
    {
        #ifdef DEBUG
            printf("Open file %s succeeds\n", f_strReadPath.c_str());
        #endif
    }
    else{
        #ifdef DEBUG
            printf("Open file %s fails\n", f_strReadPath.c_str());
        #endif
    }
    fp.read((char*)m_pEnergyProfile, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000);
    fp.close();
}

/**********************************************************
Function：         GetEnergyCorrFactor(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId)
Description：      Return an energy correction factor array for a specific Crystal
                   with information of BDM, DU, and Crystal.  
Called By：        UCoinPET.cpp
Return：           float*
Others：           None
**********************************************************/
float* UEnergyProfile::GetEnergyCorrFactor(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId)
{
    return m_pEnergyCorrFactor + f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 +
                                 f_nDUId * m_nCrystalSize * m_nCrystalSize * 1000 +
                                 f_nLocalCrystalId * 1000;
}


uint32* UEnergyProfile::GetEnergyProfile(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId)
{
    return m_pEnergyProfile + f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 +
                                 f_nDUId * m_nCrystalSize * m_nCrystalSize * 1000 +
                                 f_nLocalCrystalId * 1000;
}

uint32 UEnergyProfile::GetEnergyRecord(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId)
{
    return *(m_pEnergyRecord + f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize  +
                                 f_nDUId * m_nCrystalSize * m_nCrystalSize +
                                 f_nLocalCrystalId);
}

void UEnergyProfile::SetEnergyRecord(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId,uint32 f_nXValue)
{
    *(m_pEnergyRecord + f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize  +
                                     f_nDUId * m_nCrystalSize * m_nCrystalSize +
                                     f_nLocalCrystalId)=f_nXValue;
}

void UEnergyProfile::SetEnergyCorrFactor(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId,float f_fValue)
{
    for(uint16 j = 0; j < 1000; j++)
    {
        *(m_pEnergyCorrFactor + f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize * 1000 +
                                         f_nDUId * m_nCrystalSize * m_nCrystalSize * 1000 +
                                         f_nLocalCrystalId * 1000 + j ) = f_fValue;
    }
}
