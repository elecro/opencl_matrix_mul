#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <vector>
#include <string>

class Matrix
{
public:
    Matrix(int width, int height);
    Matrix(int width, int height, float filler);
    Matrix(const Matrix& matrix);
    Matrix(int width, int height, std::vector<float> data);


    static Matrix random(int width, int height, int limit);

    virtual int width(void) const { return m_width; }
    virtual int height(void) const { return m_height; }

    virtual float operator[](int idx) const { return m_data[idx]; }
    virtual const float *data(void) const { return &m_data[0]; }
    virtual int dataSize(void) const { return sizeof(float) * m_width * m_height; }

    Matrix transpose(void) const;

    bool operator==(const Matrix& other);
    bool operator!=(const Matrix& other) { return !this->operator==(other); }

private:
    Matrix& operator=(Matrix&);

    int m_width;
    int m_height;
    std::vector<float> m_data;
};

void print(Matrix& matrix);

#endif // _MATRIX_HPP
