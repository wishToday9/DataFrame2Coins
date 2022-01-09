/********************************************************************

  Copyright (C), 2019, All rights reserved

  File Name     :    UPositionMap.cpp
  Description   :
  History       :

  <author>            <time>            <desc>
  Lingyiqing          2019/7/10         create

********************************************************************/
#include "UAcquireDef.h"
#include "UPositionMap.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <windows.h>
#include <memory.h>
#include <malloc.h>
#include <math.h>
#include <vector>

using namespace std;
#define DEBUG

/*********************** 
 Constrctor Function 
***********************/
UPositionMap::UPositionMap(uint32 f_nBDMNum, uint32 f_nDUNum, uint32 f_nCrystalSize, uint32 f_nPositionSize)
{
    m_nBDMNum = f_nBDMNum;                //24
    m_nDUNum = f_nDUNum;                  //4
    m_nPositionSize = f_nPositionSize;    //256
    m_nCrystalSize = f_nCrystalSize;      //13

    /* Allocate memory for m_pPositionMap and m_pPositionTable, and set to zero */
    m_pPositionMap = new uint32[m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize]; //4*24*4*256*256 =24MB
    cudaHostAlloc((void**)&m_pPositionTable, m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize * sizeof(uint8), 0);
    // m_pPositionTable = new uint8[m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize];//1*24*4*256*256 =6MB
    m_pMassPoint = new Point[m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize];      //2*24*4*13*13 =32KB

    memset(m_pPositionMap, 0, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    memset(m_pPositionTable, 0, sizeof(uint8) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    memset(m_pMassPoint,0,sizeof(Point) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize);

}

/***********************
 Destructor Function 
***********************/
UPositionMap::~UPositionMap()
{
    RELEASE_POINTER(m_pPositionMap);
    cudaFreeHost(m_pPositionTable);
    //RELEASE_POINTER(m_pPositionTable);   
    RELEASE_POINTER(m_pMassPoint);
}

void UPositionMap::clear()
{
    memset(m_pPositionMap, 0, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    memset(m_pPositionTable, 0, sizeof(uint8) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    memset(m_pMassPoint,0,sizeof(Point) * m_nBDMNum * m_nDUNum * m_nCrystalSize * m_nCrystalSize);
}

/**********************************************************
Function：         CreatePositionMap(string f_strReadPath)
Description：      To create a position map for each detector unit,
                   saved in m_pPositionMap, which records the counts
                   on each pixel(256 * 256).
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UPositionMap::CreatePositionMap(string f_strReadPath)
{
    ifstream fp;
    uint32 l_nFileSize = 0;
    uint32 l_nFrameNum = 0;

    /* Read raw data from files. Counts on each pixel are accumulating */
    for(uint32 i = 0; i < m_nBDMNum; i++)
    {
        //the format of sample input file is changed
        //before
        //fp.open((f_strReadPath + to_string(i)).c_str());
        //now

        fp.open((f_strReadPath + "/thread0-channel" + to_string(i)).c_str());

        if(fp.is_open())
        {
            #ifdef DEBUG
                //printf("Open file %s succeeds\n", (f_strReadPath + to_string(i)).c_str());
                printf("Open file %s succeeds\n", (f_strReadPath + "/thread0-channel" + to_string(i)).c_str());
            #endif
        }
        else{
            #ifdef DEBUG
                //printf("Open file %s fails\n", (f_strReadPath + to_string(i)).c_str());
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
            /* Get the info of DU, BDM, X, Y */
            uint32 l_nDUId = l_pTempFrame[j].nHeadAndDU & (0x0F);
            uint32 l_nBDMId = l_pTempFrame[j].nBDM;
            uint32 l_nX = l_pTempFrame[j].X;
            uint32 l_nY = l_pTempFrame[j].Y;
            
            /* Counts are accumulating */
            m_pPositionMap[l_nBDMId * m_nDUNum * m_nPositionSize * m_nPositionSize + 
                           l_nDUId * m_nPositionSize * m_nPositionSize + 
                           l_nY * m_nPositionSize + l_nX] ++;
        }
        /* Close the file and release the memory. Start again. */
        fp.close();
        RELEASE_ARRAY_POINTER(l_pTempFrame);
    }    
}

/**********************************************************
Function：         SavePositionMap(string f_strSavePath)
Description：      To Save position maps to file, from m_pPositionMap
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UPositionMap::SavePositionMap(string f_strSavePath)
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
    fp.write((char*)m_pPositionMap, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    fp.close();

    #ifdef DEBUG
        printf("write file %s finished\n", f_strSavePath.c_str());
    #endif
}


/**********************************************************
Function：         ReadPositionMap(string f_strReadPath)
Description：      To read position maps from file, to m_pPositionMap
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UPositionMap::ReadPositionMap(string f_strReadPath)
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
    fp.read((char*)m_pPositionMap, sizeof(uint32) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    fp.close();
}


/**********************************************************
Function：         CreatePositionTable()
Description：      To create a position table for each detector 
                   unit, saved in m_pPositionTable, which 
                   records the GroupId(0-168) on each pixel(256 * 256).
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UPositionMap::CreatePositionTable()
{
	/* Allocate mass points for each detector unit */
	Point* l_pMassPoint = new Point[m_nCrystalSize * m_nCrystalSize];
    /* Allocate a temporary map and table for each detector unit */
    uint32* l_pTempMap = new uint32[m_nPositionSize * m_nPositionSize];
    uint8* l_pTempTable = new uint8[m_nPositionSize * m_nPositionSize];	
	memset(l_pTempMap, 0, sizeof(uint32) * m_nPositionSize * m_nPositionSize);
	memset(l_pTempTable, 0, sizeof(uint8) * m_nPositionSize * m_nPositionSize);
	memset(l_pMassPoint, 0, sizeof(Point) * m_nCrystalSize * m_nCrystalSize);
	
	for(uint32 i = 0; i < m_nBDMNum; i++)
    {
        for(uint32 j = 0; j < m_nDUNum; j++)
        {
            memcpy(l_pMassPoint,m_pMassPoint + i * m_nDUNum * m_nCrystalSize * m_nCrystalSize + j * m_nCrystalSize * m_nCrystalSize,sizeof(Point) * m_nCrystalSize * m_nCrystalSize);

            memcpy(l_pTempMap, m_pPositionMap + i * m_nDUNum * m_nPositionSize * m_nPositionSize + j * m_nPositionSize * m_nPositionSize, sizeof(uint32) * m_nPositionSize * m_nPositionSize);
            /*************/ 
            /* Important */
            //KMeanAlgorithm1D(l_pTempMap, l_pMassPoint);
            KMeanAlgorithm2D(l_pTempMap, l_pTempTable, l_pMassPoint);
            /*************/
            memcpy(m_pPositionTable + i * m_nDUNum * m_nPositionSize * m_nPositionSize + j * m_nPositionSize * m_nPositionSize, l_pTempTable, sizeof(uint8) * m_nPositionSize * m_nPositionSize);
           
  		    memset(l_pTempMap, 0, sizeof(uint32) * m_nPositionSize * m_nPositionSize);
			memset(l_pTempTable, 0, sizeof(uint8) * m_nPositionSize * m_nPositionSize);
			memset(l_pMassPoint, 0, sizeof(Point) * m_nCrystalSize * m_nCrystalSize);
		}
    }

#ifdef DEBUG
    printf("UPositionMap: create position table finished\n");
#endif

}

/**********************************************************
Function：         KMeanAlgorithm1D(uint32* f_pMap, Point* f_pPoint)
Description：      Get mass points from a position map with k-mean algorithm
Called By：        UPositionMap::CreatePositionTable()
Return：           None
Others：           None
**********************************************************/
void UPositionMap::KMeanAlgorithm1D(uint32* f_pMap, Point* f_pPoint)
{
	uint32* l_pTempMap = f_pMap;
	Point* l_pMassPoint = f_pPoint;
	
	/* Horizontal */
	uint32* l_pHorTempMap = new uint32[m_nPositionSize];
	memset(l_pHorTempMap, 0, sizeof(uint32)*m_nPositionSize);
	/* Accumulate in horizontal direction */
	for(uint32 x = 0; x < m_nPositionSize; x++)
	{
		for(uint32 y = 0; y < m_nPositionSize; y++)
		{
			l_pHorTempMap[x] += l_pTempMap[y*m_nPositionSize+x];
		}
	}
	/* Vertical */
	uint32* l_pVerTempMap = new uint32[m_nPositionSize];
	memset(l_pVerTempMap, 0, sizeof(uint32)*m_nPositionSize);
	/* Accumulate in vertical direction */
	for(uint32 y = 0; y < m_nPositionSize; y++)
	{
		for(uint32 x = 0; x < m_nPositionSize; x++)
		{
			l_pVerTempMap[y] += l_pTempMap[y*m_nPositionSize+x];
		}
	}
	
	/* Generate initial mass points. Interactive mode shall be involved */
	Point* l_pHorMassPoint = new Point[m_nCrystalSize];
	Point* l_pVerMassPoint = new Point[m_nCrystalSize];
    std::vector<uint8> temp{42, 48, 63, 76, 93, 109, 127, 143, 158, 176, 190, 205, 213};
	for(uint32 i = 0; i < m_nCrystalSize; i++)
	{
		l_pHorMassPoint[i].y = 0;
		l_pHorMassPoint[i].x = temp[i];
		l_pVerMassPoint[i].x = 0;
		l_pVerMassPoint[i].y = temp[i];
		#ifdef DEBUG
	        printf("initial: l_pHorMassPoint[%d].x = %d\n", i, l_pHorMassPoint[i].x);
			printf("initial: l_pVerMassPoint[%d].y = %d\n", i, l_pVerMassPoint[i].y);
	    #endif
	}
	
	/* K-mean in horizontal direction */
	uint8 l_fHorMinDist = m_nPositionSize-1;
	uint8 l_nHorMinDistMassPointId = m_nCrystalSize-1;
	PointSum* l_pHorPointSum = new PointSum[m_nCrystalSize];
	uint64* l_pHorPointNum = new uint64[m_nCrystalSize];
	memset(l_pHorPointSum, 0, sizeof(PointSum) * m_nCrystalSize);
	memset(l_pHorPointNum, 0, sizeof(uint64) * m_nCrystalSize);
	for(uint32 iter = 0; iter < 5; iter++)
	{
		for(uint32 x = 0; x < m_nPositionSize; x++)
		{
			for(uint32 k = 0; k < m_nCrystalSize; k++)
		    {
				/* Search for the minimal distance */
				if(abs(int(x-l_pHorMassPoint[k].x)) < l_fHorMinDist)
				{
					l_fHorMinDist = abs(int(x-l_pHorMassPoint[k].x));
					l_nHorMinDistMassPointId = k;
				}
			}
			l_pHorPointSum[l_nHorMinDistMassPointId].x += x * l_pHorTempMap[x];
			l_pHorPointNum[l_nHorMinDistMassPointId] += l_pHorTempMap[x];
			l_fHorMinDist = m_nPositionSize-1;
	        l_nHorMinDistMassPointId = m_nCrystalSize-1;
		}
		/* Update mass points */
		for(uint32 k = 0; k < m_nCrystalSize; k++)
		{
			if(l_pHorPointNum[k] == 0)
			{
				continue;
			}
			l_pHorMassPoint[k].x = uint8(l_pHorPointSum[k].x / l_pHorPointNum[k]);
			#ifdef DEBUG
			    printf("iter = %d, l_pHorMassPoint[%d] = %d\n", iter, k, l_pHorMassPoint[k].x);
	        #endif
		}
		memset(l_pHorPointSum, 0, sizeof(PointSum) * m_nCrystalSize);
		memset(l_pHorPointNum, 0, sizeof(uint64) * m_nCrystalSize);
	}
	
	
	/* K-mean in vertical direction */
	uint8 l_fVerMinDist = m_nPositionSize-1;
	uint8 l_nVerMinDistMassPointId = m_nCrystalSize-1;
	PointSum* l_pVerPointSum = new PointSum[m_nCrystalSize];
	uint64* l_pVerPointNum = new uint64[m_nCrystalSize];
	memset(l_pVerPointSum, 0, sizeof(PointSum) * m_nCrystalSize);
	memset(l_pVerPointNum, 0, sizeof(uint64) * m_nCrystalSize);
	for(uint32 iter = 0; iter < 5; iter++)
	{
		for(uint32 y = 0; y < m_nPositionSize; y++)
		{
			for(uint32 k = 0; k < m_nCrystalSize; k++)
		    {
				/* Search for the minimal distance */
				if(abs(int(y-l_pVerMassPoint[k].y)) < l_fVerMinDist)
				{
					l_fVerMinDist = abs(int(y-l_pVerMassPoint[k].y));
					l_nVerMinDistMassPointId = k;
				}
			}
			l_pVerPointSum[l_nVerMinDistMassPointId].y += y * l_pVerTempMap[y];
			l_pVerPointNum[l_nVerMinDistMassPointId] += l_pVerTempMap[y];
			l_fVerMinDist = m_nPositionSize-1;
	        l_nVerMinDistMassPointId = m_nCrystalSize-1;
		}
		/* Update mass points */
		for(uint32 k = 0; k < m_nCrystalSize; k++)
		{
			if(l_pVerPointNum[k] == 0)
			{
				continue;
			}
			l_pVerMassPoint[k].y = uint8(l_pVerPointSum[k].y / l_pVerPointNum[k]);
			#ifdef DEBUG
			    printf("iter = %d, l_pVerMassPoint[%d] = %d\n", iter, k, l_pVerMassPoint[k].y);
	        #endif
		}
		memset(l_pVerPointSum, 0, sizeof(PointSum) * m_nCrystalSize);
		memset(l_pVerPointNum, 0, sizeof(uint64) * m_nCrystalSize);
	}

    /* Assign for the 2D mass points */	
	for(uint32 y = 0; y < m_nCrystalSize; y++)
	{
		for(uint32 x = 0; x < m_nCrystalSize; x++)
		{
			l_pMassPoint[y*m_nCrystalSize+x].x = l_pHorMassPoint[x].x;
			l_pMassPoint[y*m_nCrystalSize+x].y = l_pVerMassPoint[y].y;
			#ifdef DEBUG
	            printf("1D: l_pMassPoint[y*m_nCrystalSize+x].x = %d\n", l_pMassPoint[y*m_nCrystalSize+x].x);
				printf("1D: l_pMassPoint[y*m_nCrystalSize+x].y = %d\n", l_pMassPoint[y*m_nCrystalSize+x].y);
	        #endif
		}
	}
	
}


/**********************************************************
Function：         KMeanAlgorithm2D(uint32* f_pMap, uint8* f_pTable, Point* f_pPoint)
Description：      Get a position table from a position map with k-mean algorithm
Called By：        UPositionMap::CreatePositionTable()
Return：           None
Others：           None
**********************************************************/
void UPositionMap::KMeanAlgorithm2D(uint32* f_pMap, uint8* f_pTable, Point* f_pPoint)
{
    uint32* l_pTempMap = f_pMap;
	uint8* l_pTempTable = f_pTable;
	Point* l_pMassPoint = f_pPoint;

    float l_fMinDist = sqrt(2)*(m_nPositionSize-1);
    uint8 l_nMinDistMassPointId = m_nCrystalSize*m_nCrystalSize-1;

    /* To calculate the weighted average of each group to get the mass point */
    PointSum* l_pPointSum = new PointSum[m_nCrystalSize * m_nCrystalSize];
    uint64* l_pPointNum = new uint64[m_nCrystalSize * m_nCrystalSize];
    memset(l_pPointSum, 0, sizeof(PointSum) * m_nCrystalSize * m_nCrystalSize);
    memset(l_pPointNum, 0, sizeof(uint64) * m_nCrystalSize * m_nCrystalSize);

    /* Processing for each map */
    for(uint32 iter = 0; iter < 5; iter++)
    {
        for(uint32 y = 0; y < m_nPositionSize; y++)
        {
            for(uint32 x = 0; x < m_nPositionSize; x++) 
            {
                /* Look for which group each pixel belong to (with the minimum distance) */
                for(uint32 k = 0; k < m_nCrystalSize * m_nCrystalSize; k++)
                {
                    if(((x-l_pMassPoint[k].x) * (x-l_pMassPoint[k].x) + (y-l_pMassPoint[k].y) * (y-l_pMassPoint[k].y)) < l_fMinDist * l_fMinDist)  
                    {
                        l_fMinDist = sqrt((x-l_pMassPoint[k].x) * (x-l_pMassPoint[k].x) + (y-l_pMassPoint[k].y) * (y-l_pMassPoint[k].y));
                        l_nMinDistMassPointId = k;
                    }
                }
				l_pTempTable[x + y * m_nPositionSize] = l_nMinDistMassPointId;
                l_pPointSum[l_nMinDistMassPointId].x += x * l_pTempMap[x + y * m_nPositionSize];
                l_pPointSum[l_nMinDistMassPointId].y += y * l_pTempMap[x + y * m_nPositionSize];
                l_pPointNum[l_nMinDistMassPointId] += l_pTempMap[x + y * m_nPositionSize];   
                l_fMinDist = sqrt(2)*(m_nPositionSize-1);
                l_nMinDistMassPointId = m_nCrystalSize*m_nCrystalSize-1;
            }
        }
        /* Calculate where is the mass point of each group */
        for(uint32 k = 0; k < m_nCrystalSize * m_nCrystalSize; k++)
        {
            if(l_pPointNum[k] == 0)
            {
                continue;
            }
            l_pMassPoint[k].x = uint8(l_pPointSum[k].x / l_pPointNum[k]);
            l_pMassPoint[k].y = uint8(l_pPointSum[k].y / l_pPointNum[k]);
        }
        memset(l_pPointSum, 0, sizeof(PointSum) * m_nCrystalSize * m_nCrystalSize);
        memset(l_pPointNum, 0, sizeof(uint64) * m_nCrystalSize * m_nCrystalSize);
    }
	
	for(uint32 y = 0; y < m_nCrystalSize; y++)
	{
		for(uint32 x = 0; x < m_nCrystalSize; x++)
		{
			#ifdef DEBUG
	            printf("2D: l_pMassPoint[y*m_nCrystalSize+x].x = %d\n", l_pMassPoint[y*m_nCrystalSize+x].x);
				printf("2D: l_pMassPoint[y*m_nCrystalSize+x].y = %d\n", l_pMassPoint[y*m_nCrystalSize+x].y);
	        #endif
		}
	}
}

/**********************************************************
Function：         SavePositionTable(string f_strSavePath)
Description：      To Save position tables to file, from m_pPositionTable
Called By：        main.cpp
Return：           None
Others：           None
**********************************************************/
void UPositionMap::SavePositionTable(string f_strSavePath)
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
    fp.write((char*)m_pPositionTable, sizeof(uint8) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    fp.close();

    #ifdef DEBUG
        printf("write file %s finished\n", f_strSavePath.c_str());
    #endif

}


/**********************************************************
Function：         ReadPositionTable(string f_strReadPath)
Description：      To Read position tables to m_pPositionTable, from file  
Called By：        UCoinPET.cpp
Return：           None
Others：           None
**********************************************************/
void UPositionMap::ReadPositionTable(string f_strReadPath)
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
    fp.read((char*)m_pPositionTable, sizeof(uint8) * m_nBDMNum * m_nDUNum * m_nPositionSize * m_nPositionSize);
    fp.close();

}

/**********************************************************
Function：         GetPositionTable(uint32 f_nBDMId, uint32 f_nDUId)
Description：      Return a position table for a specific Detector Unit
                   with information of BDM and DU.  
Called By：        UCoinPET.cpp
Return：           uint8*
Others：           None
**********************************************************/
uint8* UPositionMap::GetPositionTable(uint32 f_nBDMId, uint32 f_nDUId)
{
    return m_pPositionTable + f_nBDMId * m_nDUNum * m_nPositionSize * m_nPositionSize + f_nDUId * m_nPositionSize * m_nPositionSize;
}

/**********************************************************
Function：         GetPositionMap(uint32 f_nBDMId, uint32 f_nDUId)
Description：      Return a position map for a specific Detector Unit
                   with information of BDM and DU.
Called By：        UCoinPET.cpp
Return：           uint32*
Others：           None
**********************************************************/
uint32* UPositionMap::GetPositionMap(uint32 f_nBDMId, uint32 f_nDUId)
{
    return m_pPositionMap + f_nBDMId * m_nDUNum * m_nPositionSize * m_nPositionSize + f_nDUId * m_nPositionSize * m_nPositionSize;
}

/**********************************************************
Function：
Description：      Set a Mass Point for a specific Detector Unit.
Called By：
Return：           None
Others：           None
**********************************************************/
void UPositionMap::SetMassPoint(uint32 f_nBDMId, uint32 f_nDUId, set<uint8> f_rowSet, set<uint8> f_colSet)
{
    if(f_rowSet.size()!=m_nCrystalSize || f_colSet.size()!= m_nCrystalSize)
    {
        #ifdef DEBUG
            printf("UPositionMap: row or col set size is wrong, row set size:%d  col set size:%d\n",(int)f_rowSet.size(),(int)f_colSet.size());
        #endif
        return;
    }

    //row is y, col is x
    set<uint8>::iterator iterRowY;
    set<uint8>::iterator iterColX;

    uint32 l_nBase=f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize + f_nDUId * m_nCrystalSize * m_nCrystalSize;

    for(iterRowY=f_rowSet.begin();iterRowY!=f_rowSet.end();iterRowY++)
    {
        for(iterColX=f_colSet.begin();iterColX!=f_colSet.end();iterColX++)
        {
            m_pMassPoint[l_nBase].y=*iterRowY;
            m_pMassPoint[l_nBase].x=*iterColX;
            l_nBase++;
        }
    }

    #ifdef DEBUG
        int i=f_nBDMId * m_nDUNum * m_nCrystalSize * m_nCrystalSize + f_nDUId * m_nCrystalSize * m_nCrystalSize;
        for(int j=0;j<13*13;j++)
        {
            printf("UPositionMap: BDM:%d,  DU:%d mass point y:%d,  x:%d\n",f_nBDMId,f_nDUId,m_pMassPoint[i+j].y,m_pMassPoint[i+j].x);
        }
    #endif
}















