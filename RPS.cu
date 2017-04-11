#include "RPS.h"

class RPSSimImpl
{
public:
	RPSSimImpl(const char *strInitStatePath)
	{

	}

	void MakeOneRPSFrame(void *surface, size_t width, size_t height, size_t pitch, float t)
	{

	}

	virtual ~RPSSimImpl() {}
};

RPSSim::RPSSim(const char *strInitStatePath)
{
	m_impl = new RPSSimImpl(strInitStatePath);
}

RPSSim::~RPSSim()
{
	delete m_impl;
}

void RPSSim::MakeOneRPSFrame(void *surface, size_t width, size_t height, size_t pitch, float t)
{
	m_impl->MakeOneRPSFrame(surface, width, height, pitch, t);
}
