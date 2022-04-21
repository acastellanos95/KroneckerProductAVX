//
// Created by andre on 4/20/22.
//

#ifndef KRONECKERPRODUCT__MATRIX_H_
#define KRONECKERPRODUCT__MATRIX_H_

#include <vector>

template<typename T>
struct Matrix{
  unsigned long width;
  unsigned long height;
  std::vector<T> elements;
};

#endif //KRONECKERPRODUCT__MATRIX_H_
