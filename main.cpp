#include <iostream>
#include "Utils.h"

int main() {
  // Example correct
  Matrix<float> A;

  Matrix<float> B;

  // Example correct
  A.width = 3;
  A.height = 2;
  A.elements = std::vector<float> {1.0, -4.0, 7.0, -2.0, 3.0, 3.0};

  B.width = 4;
  B.height = 4;
  B.elements = std::vector<float> {8.0, -9.0, -6.0, 5.0, 1.0, -3.0, -4.0, 7.0, 2.0, 8.0, -8.0, -3.0, 1.0, 2.0, -5.0, -1.0};

  auto C = normalKroneckerMultiplication(A, B);
  auto CIntrinsics = intrinsicsNormalKroneckerMultiplication(A, B);
  printMatrix(C);
  printMatrix(CIntrinsics);

  // Random
  auto ARandom = Matrix<float>();
  ARandom.height = 30;
  ARandom.width = 50;
  ARandom.elements = std::vector<float>(ARandom.width*ARandom.height,0.0);
  initMatrix(ARandom);
  auto BRandom = Matrix<float>();
  BRandom.height = 500;
  BRandom.width = 500;
  BRandom.elements = std::vector<float>(BRandom.width*BRandom.height,0.0);
  initMatrix(BRandom);

  auto ti = clock();
  C = normalKroneckerMultiplication(ARandom, BRandom);
  auto tf = clock();
  auto time = (((float)tf - (float)ti) / CLOCKS_PER_SEC );
  std::cout << "Tiempo normal producto de Kronecker: " << std::to_string(time) << '\n';

  ti = clock();
  CIntrinsics = intrinsicsNormalKroneckerMultiplication(ARandom, BRandom);
  tf = clock();
  time = (((float)tf - (float)ti) / CLOCKS_PER_SEC );
  std::cout << "Tiempo instrinsic producto de Kronecker: " << std::to_string(time) << '\n';

  float maxError = 0.0;
  for (size_t rowIndex = 0; rowIndex < C.height; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < C.width; ++colIndex) {
      maxError = std::max(std::abs(C.elements[rowIndex*C.width + colIndex] - CIntrinsics.elements[rowIndex*C.width + colIndex]), maxError);
    }
  }
  std::cout << "Mayor error: " << std::to_string(maxError) << '\n';

  return 0;
}
