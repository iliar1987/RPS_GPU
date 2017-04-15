#pragma once


class RPSSim
{
private:
	float* m_d_lastFrame;
	float* m_d_thisFrame;
	float *m_d_arrScores;
	int m_width, m_height;
	size_t m_pitch;
	int RandomizeBuffer(float *d_buffer);

public:
	int GetWidth() const { return m_width; }
	int GetHeight() const { return m_height; }
	size_t GetPitch() const { return m_pitch; }
	RPSSim(const char *strInitStatePath);

	RPSSim(int width, int height);

	void* MakeOneRPSFrame(float t, const float fNormPower, const float fDiffusionPower, const float fDiffusionCoeff);

	virtual ~RPSSim();
};