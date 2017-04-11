#pragma once

class RPSSimImpl;
class RPSSim
{
	RPSSimImpl* m_impl = nullptr;
public:
	void MakeOneRPSFrame(void *surface, size_t width, size_t height, size_t pitch, float t);
	RPSSim(const char* strInitStatePath);
	virtual ~RPSSim();
};
