#ifndef UNTIL
#define UNTIL

unsigned nextPowerOf2(unsigned value)
{
	if ((value != 0) && ((value & (value - 1)) == 0))
	{
		return value;
	}

	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value++;

	return value;
}

unsigned previousPowerOf2(unsigned value) {
	if ((value != 0) && ((value & (value - 1)) == 0))
	{
		return value;
	}

	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value -= value >> 1;

	return value;
}

int roundUp(int numToRound, int multiple)
{
	if (multiple == 0)
	{
		return numToRound;
	}

	int remainder = numToRound % multiple;

	if (remainder == 0)
	{
		return numToRound;
	}

	return numToRound + multiple - remainder;
}

#endif // !UNTIL

#pragma once
