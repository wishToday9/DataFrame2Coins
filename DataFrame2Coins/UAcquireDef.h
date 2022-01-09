/********************************************************************

  Copyright (C), 2019, All rights reserved

  File Name     :    UAcquireDef.h
  Description   :
  History       :

  <author>            <time>            <desc>
  Lingyiqing          2019/1/1          create

********************************************************************/

#ifndef __UACQUIREDEF_H
#define __UACQUIREDEF_H

#include "UTypeDef.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include <string>

#pragma pack (push, 1)

/* Frame received from BDM */
typedef struct _DataFrameV2
{
    uint8    nHeadAndDU;
    uint8    nBDM;
    uint8    nTime[8];
    uint8    X;
    uint8    Y;
    uint8    Energy[2];
    int8     nTemperatureInt;
    uint8    nTemperatureAndTail;
}DataFrameV2;

/* Used in the old version */
typedef struct _ResolveFrame
{
    uint64   nTime;
    uint8    X;
    uint8    Y;
    uint16   nEnergy;
    uint8    PMT;
    uint8    BDM;
    float    fTemperature;
}ResolveFrame;

/* Used in clinical PET */
typedef struct _SamplesStruct
{
    uint16    globalBDMIndex;
    uint16    localCrystalIndex;
    double    timevalue[8];
}SamplesStruct;

/* Temp structure. Local -> Global */
typedef struct _TempSinglesStruct
{
    uint16    globalBDMIndex;
    uint16    localDUIndex;
    uint16    localCrystalIndex;
    float     energy;
    double    timevalue;
}TempSinglesStruct;

/* Used in the new version */
typedef struct _SinglesStruct
{
    uint32    globalCrystalIndex;
    float     energy;
    double    timevalue;
}SinglesStruct;

/* Used in listmode */
typedef struct _CoinStruct
{
    SinglesStruct nCoinStruct[2];
}CoinStruct;

#pragma pack(pop)


/* The size of crystal in one DU */
//const uint32 CRYSTAL_SIZE = 13;
/* The size of position in one DU */
//const uint32 POSITION_SIZE = 256;
/* The num of DU */
//const uint32 DU_NUM = 4;
/* The num of BDM */
//const uint32 BDM_NUM = 24;
/* The length of time window */
//const uint32 TIME_WINDOW = 5000;
/*************************************/
/* The minimum energy */
//const float MIN_ENERGY = 350.0;
/* The maximum energy */
//const float MAX_ENERGY = 650.0;
/*************************************/
/* The num of a common data buffer, 150MB for DataFrameV2 */
//const uint32 FRAME_NUM_ONE_BUFFER = 64 * 1024 * 150 ;
/* The num of total packages in one channel, 150MB */
//const uint32 PACKAGE_NUM = 1024 * 150;
/* The size of one package, 1KB */
//const uint32 PACKAGE_SIZE = 1024;
/* The size of one frame */
//const uint32 FRAME_SIZE = sizeof(DataFrameV2);
/* The num of frames in one package */
//const uint32 FRAME_NUM_ONE_PKG = PACKAGE_SIZE / FRAME_SIZE;
/* The version of the udp frame */
//const uint32 UDP_FRAME_VERSION = 2;

/* To save preset position map */
//static const string strPositionMapPath = "../Preset/PositionMap.dat";
/* To save preset position table */
//static const string strPositionTablePath = "../Preset/PositionTable.dat";
/* To save preset energy profile */
//static const string strEnergyProfilePath = "../Preset/EnergyProfile.dat";
/* To save preset energy correction factor */
//static const string strEnergyCorrFactorPath = "../Preset/EnergyCorrFactor.dat";

#endif

