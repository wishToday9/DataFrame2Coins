/********************************************************************

  Copyright (C), 2019, All rights reserved

  File Name     :    UEnergyProfile.h
  Description   :
  History       :

  <author>            <time>            <desc>
  Lingyiqing          2019/7/22         create

********************************************************************/
#ifndef __UENERGYPROFILE_H
#define __UENERGYPROFILE_H

#include <string>
#include "UAcquireDef.h"
#include "UTypeDef.h"
#include "UPositionMap.h"

class UPositionMap;

class UEnergyProfile
{
public:
    UEnergyProfile(uint32 f_nBDMNum, uint32 f_nDUNum, uint32 f_nCrystalSize, uint32 f_nPositionSize);
    virtual        ~UEnergyProfile();

    /* Create an energy profile for each Crystal. Saved in m_pEnergyProfile */
    void           CreateEnergyProfile(std::string f_strReadPath, std::string f_strPositionTablePath);
    /* Save energy profiles to file, from m_pEnergyProfile */
    void           SaveEnergyProfile(std::string f_strSavePath);

    /* Create an energy correction factor array for each Crystal. Saved in m_pEnergyCorrFactor */
    void           CreateEnergyCorrFactor();
    /* Save energy correction factor arrays to file, from m_pEnergyCorrFactor */
    void           SaveEnergyCorrFactor(std::string f_strSavePath);
    
    /* Read energy correction factor arrays to m_pEnergyCorrFactor, from file */
    void           ReadEnergyCorrFactor(std::string f_strReadPath);
    /* Read energy profile arrays to m_pEnergyProfile, from file */
    void           ReadEnergyProfile(std::string f_strReadPath);
    /* Return an energy correction factor array for a specific Crystal */
    float*         GetEnergyCorrFactor(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId);
    /* Return an energy profile array for a specific Crystal */
    uint32*        GetEnergyProfile(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId);
    /* Set an energy correction factor array for a specific Crystal */
    void           SetEnergyCorrFactor(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId,float f_fValue);
    /*memset all array to  0 */
    void           Clear();
    /* Return an energy record of x coordinate for a specific Crystal */
    uint32         GetEnergyRecord(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId);
    /* Set an energy record of x coordinate for a specific Crystal */
    void           SetEnergyRecord(uint32 f_nBDMId, uint32 f_nDUId, uint32 f_nLocalCrystalId,uint32 f_nXValue);

private:
    /* Get an energy correction factor array from an energy profile */
    void           NormalizeEnergy(uint32* f_pProfile, float* f_pCorrFactor);

    UPositionMap*  m_pPositionMap;

    /* Number of BDM */
    uint32         m_nBDMNum;
    /* Number of DU */
    uint32         m_nDUNum;
    /* Size of Crystal array */
    uint32         m_nCrystalSize;
    /* Size of position map */
    uint32         m_nPositionSize;

    /* Pointer of energy profile. Record the counts in each energy interval */
    uint32*        m_pEnergyProfile;
    /* Pointer of energy correction factor array. Record factors between 0 and 1 */
    float*         m_pEnergyCorrFactor;
    /* Pointer of energy record. Record x coordinate of max counts near 51.1 between 0 and 1000 */
    uint32*        m_pEnergyRecord;

};

#endif
