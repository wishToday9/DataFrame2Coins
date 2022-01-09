/********************************************************************

  Copyright (C), 2019, All rights reserved

  File Name     :    UPositionMap.h
  Description   :
  History       :

  <author>            <time>            <desc>
  Lingyiqing          2019/7/10         create

***************************************************source/midware/calibrationMidware/position*****************/
#ifndef __UPOSITIONMAP_H
#define __UPOSITIONMAP_H

#include <set>
#include "UAcquireDef.h"
#include "UTypeDef.h"
using namespace std;

typedef struct _Point
{
    uint8 x;
    uint8 y;
}Point;

typedef struct _PointSum
{
    uint32 x;
    uint32 y;
}PointSum;

class UPositionMap
{
public:
    UPositionMap(uint32 f_nBDMNum, uint32 f_nDUNum, uint32 f_nCrystalSize, uint32 f_nPositionSize);
    virtual     ~UPositionMap();

    /* Create a position map for each Detector Unit. Saved in m_pPositionMap */
    void        CreatePositionMap(string f_strReadPath);
    /* Save position maps to file, from m_pPositionMap */ 
    void        SavePositionMap(string f_strSavePath);
	/* Read position map to m_pPositionMap, from file */
    void        ReadPositionMap(string f_strReadPath);

    /* Create a position table for each Detector Unit. Saved in m_pPositionTable */
    void        CreatePositionTable();
    /* Save position tables to file, from m_pPositionTable */
    void        SavePositionTable(string f_strSavePath);
    /* Read position tables to m_pPositionTable, from file */
    void        ReadPositionTable(string f_strReadPath);
    /* Return a position table for a specific Detector Unit */
    uint8*      GetPositionTable(uint32 f_nBDMId, uint32 f_nDUId);
    /* Return a position map for a specific Detector Unit */
    uint32*     GetPositionMap(uint32 f_nBDMId, uint32 f_nDUId);
    /*Set a Mass Point for a specific Detector Unit */
    void        SetMassPoint(uint32 f_nBDMId, uint32 f_nDUId,set<uint8> f_rowSet,set<uint8> f_colSet);
    /*memset m_pPositionMap ,m_pPositonTable and m_pMassPoint  0 */
    void        clear();

private:
    /* Get a 2D position table from a position map with k-mean algorithm */
	void        KMeanAlgorithm1D(uint32* f_pMap, Point* f_pPoint);   
    /* Get a 2D position table from a position map with k-mean algorithm */
    void        KMeanAlgorithm2D(uint32* f_pMap, uint8* f_pTable, Point* f_pPoint);

    /* Number of BDM */
    uint32      m_nBDMNum;
    /* Number of DU */
    uint32      m_nDUNum;
    /* Size of position map */
    uint32      m_nPositionSize;
    /* Size of crystal array */
    uint32      m_nCrystalSize;

    /* Pointer of position map. Record the counts on each pixel */
    uint32*     m_pPositionMap;
    /* Pointer of position table. Record the GroupId(0-168) on each pixel */
    uint8*      m_pPositionTable;
    /* Pointer of mass point. Record the mass points of each detector unit*/
    Point*      m_pMassPoint;

};

#endif

