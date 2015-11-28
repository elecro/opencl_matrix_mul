#include "matrix.hpp"

#include <cstdio>
#include <cstdlib>

Matrix::Matrix(int width, int height)
    : m_width(width)
    , m_height(height)
    , m_data(width * height)
{
}

Matrix::Matrix(int width, int height, float filler)
    : m_width(width)
    , m_height(height)
    , m_data(width * height, filler)
{
}

Matrix::Matrix(const Matrix& matrix)
    : m_width(matrix.m_width)
    , m_height(matrix.m_height)
    , m_data(matrix.m_data)
{
}

Matrix::Matrix(int width, int height, std::vector<float> data)
    : m_width(width)
    , m_height(height)
    , m_data(data)
{
}


Matrix Matrix::random(int width, int height, int limit)
{
    std::vector<float> data(width * height);

    for (int i = 0; i < width * height; i++)
    {
        data[i] = rand() % limit;
    }

    return Matrix(width, height, data);
}

Matrix Matrix::transpose(void) const
{
    int width = m_height;
    int height = m_width;
    std::vector<float> data(width * height);

    for (int y = 0; y < m_height; y++)
    {
        for (int x = 0; x < m_width; x++)
        {
            data[x * width + y] = m_data[y * m_width + x];
        }
    }

    return Matrix(width, height, data);
}

bool Matrix::operator==(const Matrix& other)
{
    if (m_width != other.width() || m_height != other.height())
    {
        return false;
    }

    int size = m_width * m_height;
    for (int i = 0; i < size; i++)
    {
        if (m_data[i] != other[i])
        {
            return false;
        }
    }

    return true;
}

void print(Matrix& matrix)
{
    int height = matrix.height();
    int width = matrix.width();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%.2f ", matrix[y * width + x]);
        }
        printf("\n");
    }
}
