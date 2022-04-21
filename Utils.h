//
// Created by andre on 4/20/22.
//

#ifndef KRONECKERPRODUCT__UTILS_H_
#define KRONECKERPRODUCT__UTILS_H_

#include <vector>
#include <random>
#include <iostream>
#include <immintrin.h>
#include "Matrix.h"

template<typename T>
std::vector<T> flatMatrix(std::vector<std::vector<T>> &matrix) {
  auto result = std::vector<T>();
  for (auto &row: matrix) {
    for (auto &element: row) {
      result.push_back(element);
    }
  }
  return result;
}

template<typename T>
void initMatrix(Matrix<T> &A){
  if(std::is_integral<T>::value){
    std::random_device rd;
    std::default_random_engine e( rd() );
    std::uniform_int_distribution<long> uniform_dist(0, 5);
    for(size_t indexRow = 0; indexRow < A.height; ++indexRow){
      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
        A.elements[indexRow*A.width + indexCol] = uniform_dist(e);
      }
    }
  } else if(std::is_floating_point<T>::value){
    std::random_device rd;
    std::default_random_engine e( rd() );
    std::uniform_real_distribution<float> uniform_dist(0, 5);
    for(size_t indexRow = 0; indexRow < A.height; ++indexRow){
      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
        A.elements[indexRow*A.width + indexCol] = uniform_dist(e);
      }
    }
  }
}

template<typename T>
void printMatrix(Matrix<T> &A){
  for(size_t row = 0; row < A.height; ++row){
    for(size_t col = 0; col < A.width; ++col){
      std::cout << std::to_string(A.elements[row*A.width + col]) << " \n"[col == A.width - 1];
    }
  }
  std::cout << '\n';
}

template<typename T>
std::vector<T> scalarNormalMultiplicationMatrix(T constant, std::vector<T> &M){
  auto matrix = std::vector<T>(M);
  for(auto &element: matrix){
    element = constant * element;
  }
  return matrix;
}

std::vector<float> scalarIntrinsicsMultiplicationMatrix(float constant, std::vector<float> &M) {
  auto matrix = std::vector<float>(M.size(),0.0);
  auto constantVector = std::vector<float>(8, constant);
  auto scratchpad = std::vector<float>(8, 0.0);
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
      ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
  const auto col = M.size();
  size_t j = 0;
  // Ir multiplicando tratando de usar la mayor cantidad de registros YMM disponibles
  while (j < col) {
    if (j + 120 > col) {
      break;
    } else {
      ymm0 = __builtin_ia32_loadups256(&M[j]);
      ymm1 = __builtin_ia32_loadups256(&M[j + 8]);
      ymm2 = __builtin_ia32_loadups256(&M[j + 16]);
      ymm3 = __builtin_ia32_loadups256(&M[j + 24]);
      ymm4 = __builtin_ia32_loadups256(&M[j + 32]);
      ymm5 = __builtin_ia32_loadups256(&M[j + 40]);
      ymm6 = __builtin_ia32_loadups256(&M[j + 48]);
      ymm7 = __builtin_ia32_loadups256(&M[j + 56]);
      ymm8 = __builtin_ia32_loadups256(&M[j + 64]);
      ymm9 = __builtin_ia32_loadups256(&M[j + 72]);
      ymm10 = __builtin_ia32_loadups256(&M[j + 80]);
      ymm11 = __builtin_ia32_loadups256(&M[j + 88]);
      ymm12 = __builtin_ia32_loadups256(&M[j + 96]);
      ymm13 = __builtin_ia32_loadups256(&M[j + 104]);
      ymm14 = __builtin_ia32_loadups256(&M[j + 112]);
      ymm15 = __builtin_ia32_loadups256(&constantVector[0]);

      ymm0 = _mm256_mul_ps(ymm0, ymm15);
      ymm1 = _mm256_mul_ps(ymm1, ymm15);
      ymm2 = _mm256_mul_ps(ymm2, ymm15);
      ymm3 = _mm256_mul_ps(ymm3, ymm15);
      ymm4 = _mm256_mul_ps(ymm4, ymm15);
      ymm5 = _mm256_mul_ps(ymm5, ymm15);
      ymm6 = _mm256_mul_ps(ymm6, ymm15);
      ymm7 = _mm256_mul_ps(ymm7, ymm15);
      ymm8 = _mm256_mul_ps(ymm8, ymm15);
      ymm9 = _mm256_mul_ps(ymm9, ymm15);
      ymm10 = _mm256_mul_ps(ymm10, ymm15);
      ymm11 = _mm256_mul_ps(ymm11, ymm15);
      ymm12 = _mm256_mul_ps(ymm12, ymm15);
      ymm13 = _mm256_mul_ps(ymm13, ymm15);
      ymm14 = _mm256_mul_ps(ymm14, ymm15);

      // Insert to matrix
      __builtin_ia32_storeups256(&matrix[j], ymm0);
      __builtin_ia32_storeups256(&matrix[j + 8], ymm1);
      __builtin_ia32_storeups256(&matrix[j + 16], ymm2);
      __builtin_ia32_storeups256(&matrix[j + 24], ymm3);
      __builtin_ia32_storeups256(&matrix[j + 32], ymm4);
      __builtin_ia32_storeups256(&matrix[j + 40], ymm5);
      __builtin_ia32_storeups256(&matrix[j + 48], ymm6);
      __builtin_ia32_storeups256(&matrix[j + 56], ymm7);
      __builtin_ia32_storeups256(&matrix[j + 64], ymm8);
      __builtin_ia32_storeups256(&matrix[j + 72], ymm9);
      __builtin_ia32_storeups256(&matrix[j + 80], ymm10);
      __builtin_ia32_storeups256(&matrix[j + 88], ymm11);
      __builtin_ia32_storeups256(&matrix[j + 96], ymm12);
      __builtin_ia32_storeups256(&matrix[j + 104], ymm13);
      __builtin_ia32_storeups256(&matrix[j + 112], ymm14);
      j += 120;
    }
  }
  while (j < col) {
    if (j + 72 > col) {
      break;
    } else {
      ymm0 = __builtin_ia32_loadups256(&M[j]);
      ymm1 = __builtin_ia32_loadups256(&M[j + 8]);
      ymm2 = __builtin_ia32_loadups256(&M[j + 16]);
      ymm3 = __builtin_ia32_loadups256(&M[j + 24]);
      ymm4 = __builtin_ia32_loadups256(&M[j + 32]);
      ymm5 = __builtin_ia32_loadups256(&M[j + 40]);
      ymm6 = __builtin_ia32_loadups256(&M[j + 48]);
      ymm7 = __builtin_ia32_loadups256(&M[j + 56]);
      ymm8 = __builtin_ia32_loadups256(&M[j + 64]);
      ymm9 = __builtin_ia32_loadups256(&constantVector[0]);

      ymm0 = _mm256_mul_ps(ymm0, ymm9);
      ymm1 = _mm256_mul_ps(ymm1, ymm9);
      ymm2 = _mm256_mul_ps(ymm2, ymm9);
      ymm3 = _mm256_mul_ps(ymm3, ymm9);
      ymm4 = _mm256_mul_ps(ymm4, ymm9);
      ymm5 = _mm256_mul_ps(ymm5, ymm9);
      ymm6 = _mm256_mul_ps(ymm6, ymm9);
      ymm7 = _mm256_mul_ps(ymm7, ymm9);
      ymm8 = _mm256_mul_ps(ymm8, ymm9);

      // Insert to matrix
      __builtin_ia32_storeups256(&matrix[j], ymm0);
      __builtin_ia32_storeups256(&matrix[j + 8], ymm1);
      __builtin_ia32_storeups256(&matrix[j + 16], ymm2);
      __builtin_ia32_storeups256(&matrix[j + 24], ymm3);
      __builtin_ia32_storeups256(&matrix[j + 32], ymm4);
      __builtin_ia32_storeups256(&matrix[j + 40], ymm5);
      __builtin_ia32_storeups256(&matrix[j + 48], ymm6);
      __builtin_ia32_storeups256(&matrix[j + 56], ymm7);
      __builtin_ia32_storeups256(&matrix[j + 64], ymm8);
      j += 72;
    }
  }
  while (j < col) {
    if (j + 8 > col) {
      break;
    } else {
      ymm0 = __builtin_ia32_loadups256(&M[j]);
      ymm1 = __builtin_ia32_loadups256(&constantVector[0]);

      ymm0 = _mm256_mul_ps(ymm0, ymm1);

      // Insert to matrix
      __builtin_ia32_storeups256(&matrix[j], ymm0);
      j += 8;
    }
  }
  while(j < col) {
    matrix[j] = constant * M[j];
    ++j;
  }
  return matrix;
}

Matrix<float> normalKroneckerMultiplication(Matrix<float> &A, Matrix<float> &B) {
  auto C = Matrix<float>();
  C.height = A.height * B.height;
  C.width = A.width * B.width;
  C.elements = std::vector<float>(C.height * C.width, 0.0);
  auto matrices = std::vector<Matrix<float>>();

  for(size_t row = 0; row < A.height; ++row){
    for(size_t col = 0; col < A.width; ++col){
      Matrix<float> tmp;
      tmp.width = B.width;
      tmp.height = B.height;
      tmp.elements = scalarNormalMultiplicationMatrix(A.elements[row*A.width + col], B.elements);
      matrices.push_back(tmp);
    }
  }

  size_t matrixIndex = 0;
  for(size_t row = 0; row < C.height; row += B.height){
    for(size_t col = 0; col < C.width; col += B.width){
      //Copy to C
      for(size_t rowSubMatrix = 0; rowSubMatrix < B.height; ++rowSubMatrix){
        for(size_t colSubMatrix = 0; colSubMatrix < B.width; ++colSubMatrix){
          C.elements[(row + rowSubMatrix)*C.width + (col + colSubMatrix)] = matrices[matrixIndex].elements[rowSubMatrix*matrices[matrixIndex].width + colSubMatrix];
        }
      }
      ++matrixIndex;
    }
  }

  return C;
}

Matrix<float> intrinsicsNormalKroneckerMultiplication(Matrix<float> &A, Matrix<float> &B) {
  auto C = Matrix<float>();
  C.height = A.height * B.height;
  C.width = A.width * B.width;
  C.elements = std::vector<float>(C.height * C.width, 0.0);
  auto matrices = std::vector<Matrix<float>>();

  for(size_t row = 0; row < A.height; ++row){
    for(size_t col = 0; col < A.width; ++col){
      Matrix<float> tmp;
      tmp.width = B.width;
      tmp.height = B.height;
      tmp.elements = scalarIntrinsicsMultiplicationMatrix(A.elements[row*A.width + col], B.elements);
      matrices.push_back(tmp);
    }
  }

  size_t matrixIndex = 0;
  for(size_t row = 0; row < C.height; row += B.height){
    for(size_t col = 0; col < C.width; col += B.width){
      //Copy to C
      for(size_t rowSubMatrix = 0; rowSubMatrix < B.height; ++rowSubMatrix){
        for(size_t colSubMatrix = 0; colSubMatrix < B.width; ++colSubMatrix){
          C.elements[(row + rowSubMatrix)*C.width + (col + colSubMatrix)] = matrices[matrixIndex].elements[rowSubMatrix*matrices[matrixIndex].width + colSubMatrix];
        }
      }
      ++matrixIndex;
    }
  }

  return C;
}

#endif //KRONECKERPRODUCT__UTILS_H_
