/********************************************************************

Copyright (C), 2019, All rights reserved

File Name     :    PTypeDef.h
Description   :
History       :

<author>            <time>            <desc>
Lingyiqing          2019/1/1          create

********************************************************************/

#ifndef __TYPEDEF_H
#define __TYPEDEF_H

//********************************************
//			宏定义区分操作系统
//********************************************
#ifdef _WIN32//define something for Windows (32-bit and 64-bit, this part is common)
typedef unsigned char             BYTE;
typedef unsigned short            WORD;
typedef unsigned long long int    ULONGLONG;
#ifdef _WIN64//define something for Windows (64-bit only)
#else //define something for Windows (32-bit only)
#endif
#elif __linux__//linux
typedef unsigned char             BYTE;
typedef unsigned short            WORD;
typedef unsigned int              DWORD;
typedef unsigned long long int    ULONGLONG;
#elif __unix__
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
// iOS Simulator
#elif TARGET_OS_IPHONE
// iOS device
#elif TARGET_OS_MAC
// Other kinds of Mac OS
#else
#   error "Unknown Apple platform"
#endif
#else
#   error "Unknown Apple platform"
#endif

#ifndef NULL
#define NULL                nullptr
#endif

typedef unsigned char             uint8;
typedef unsigned short            uint16;
typedef unsigned int              uint32;
typedef unsigned long long int    uint64;
typedef signed char               int8;
typedef short                     int16;
typedef int                       int32;
typedef signed long long int      int64;


#define RELEASE_POINTER(p) if(p) {delete p; p = nullptr;}

#define RELEASE_ARRAY_POINTER(p) if(p) {delete [] p; p = nullptr;}

#define OFFSET_ADDR(STRUCTURE,FIELD) ((size_t)(&((STRUCTURE*)nullptr)->FIELD))


#endif
