#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace py = pybind11;


// m1, m2: mxn
// res must be allocated space first
template<typename T>
void Matrix_Sub(const T* m1, const T* m2, T* res, 
                int m, int n) 
{

  //*std::cout << "Start Matrix_Sub" << std::endl;
  memset((void*)res, 0, sizeof(T)*m*n); // memset 0 to float also valid

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
        res[i*n + j] = m1[i*n + j] - m2[i*n + j];
}

// m1: mxd, m2: dxn
// res must be allocated space first
template<typename T>
void Matrix_Mul(const T* m1, const T* m2, T* res, 
                int m, int d, int n) 
{
  //*std::cout << "Start Matrix_Mul" << std::endl;
  memset((void*)res, 0, sizeof(T)*m*n); // memset 0 to float also valid

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < d; k++) {
        res[i*n + j] += m1[i*d + k] * m2[k*n + j];
      }
}

// ma: mxn, res: nxm 
template<typename T>
void Matrix_Transpose(const T* ma, T* res, int m, int n) 
{
  //*std::cout << "Start Matrix_Transpose" << std::endl;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
        ////*std::cout << "i:" << i << ", j:" << j << std::endl;
        ////*std::cout << "a" << std::endl;
        //T a = res[j][i];
        ////*std::cout << "b" << std::endl;
        //T b = ma[i][j];
        ////*std::cout << "c" << std::endl;
        res[j*m + i] = ma[i*n + j];
    }
  //*std::cout << "End Matrix_Transpose" << std::endl;
}

// ma: mxn
// res must be allocated space first
template<typename T>
void Softmax(const T* ma, T* res, int m, int n) 
{
  //*std::cout << "Start Softmax" << std::endl;
  T* ma_exp = (T*)malloc(sizeof(T)*m*n);
  T* ma_exp_sum = (T*)malloc(sizeof(T)*m*n);

  // calculate exp of ma
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) 
      ma_exp[i*n + j] = exp(ma[i*n + j]);
    
  // calculate row-sum of exp of ma
  memset((void*)ma_exp_sum, 0, sizeof(T)*m*n); // memset 0 to float also valid
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) 
      ma_exp_sum[i*n] += ma_exp[i*n + j];

    for (int j = 1; j < n; j++) 
      ma_exp_sum[i*n + j] = ma_exp_sum[i*n];
  }

  // calculate normalized matrix 
  for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++) 
      res[i*n + j] = ma_exp[i*n + j] / ma_exp_sum[i*n + j];

  // release space
  free(ma_exp);
  free(ma_exp_sum);
}

// param -= lr / batch * X_T @ (Z - I_y)
// X_T: n x batch
// Z: batch x k
// n: feature num, k: output logits num
template<typename T>
void UpdateParam(T* param, const T* X_T, const T* Z, const T* I_y, 
                int n, int k, 
                float lr, int batch) 
{
  //*std::cout << "Start UpdateParam" << std::endl;
  float scale = lr / batch;
  T* Z_minus_Iy= (T*)malloc(sizeof(T)*batch*k);
  T* Matrix_tmp = (T*)malloc(sizeof(T)*n*k);

  for (int i = 0; i < batch; i++) 
    for (int j = 0; j < k; j++) 
      Z_minus_Iy[i*k + j] = Z[i*k + j] - I_y[i*k + j];

  Matrix_Mul((const T*)X_T, (const T*)Z_minus_Iy, Matrix_tmp, n, batch, k);

  for (int i = 0; i < n; i++) 
    for (int j = 0; j < k; j++) 
      param[i*k + j] -= scale * Matrix_tmp[i*k + j];

  // release space
  free(Z_minus_Iy);
  free(Matrix_tmp);
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    //*std::cout << "start softmax_regression_epoch_cpp" << std::endl;
    int iter_num = m / batch;

    // allocate space for some intermidiate matrix
    float* X_batch_T = (float*)malloc(sizeof(float)*batch*n);
    float* X_batch_theta = (float*)malloc(sizeof(float)*batch*k);
    float* Z_batch = (float*)malloc(sizeof(float)*batch*k);
    float* I_y = (float*)malloc(sizeof(float)*batch*k);

    for (int i = 0; i < iter_num; i++) {
      std::cout << "Start iterate [" << i + 1 << "]" << std::flush;
      const float* X_batch = (const float*)(X + batch*n*i); 
      Matrix_Transpose(X_batch, X_batch_T, batch, n);

      Matrix_Mul(X_batch, (const float*)theta, X_batch_theta, batch, n, k);
      Softmax((const float*)X_batch_theta, Z_batch, batch, k);

      memset((void*)I_y, 0, sizeof(float)*batch*k);
      for (size_t j = 0; j < batch; j++) {
        I_y[j*k + unsigned(y[batch*i+j])] = 1.0;
      }
      
      UpdateParam((float*)theta, (const float*)X_batch_T, (const float*)Z_batch, (const float*)I_y, n, k, lr, batch);
    }

    // release space
    free(X_batch_T);
    free(X_batch_theta);
    free(Z_batch);
    free(I_y);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
