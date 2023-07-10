#include <iostream>
#include <bandicoot>

using namespace std;
using namespace coot;

// Bandicoot documentation is available at:
// http://coot.sourceforge.net/docs.html

// NOTE: the C++11 "auto" keyword is not recommended for use with Bandicoot objects and functions

int
main(int argc, char** argv)
  {
  cout << "Bandicot version: " << coot_version::as_string() << endl;

  // construct a GPU matrix according to given size and set to zeros
  // GPU matrices tend to only show speedup for large matrices;
  // this example matrix is small for the sake of demonstration
  //
  // NOTE: double-precision floating point matrices (`mat`) may not be supported by all GPUs,
  // and may not show as much speedup on GPUs as single-precision floating point, so this example uses `fmat`
  fmat A(3,4);
  A.zeros();

  // .n_rows and .n_cols are read only
  cout << "A.n_rows: " << A.n_rows << endl;
  cout << "A.n_cols: " << A.n_cols << endl;

  // access an element (indexing starts at 0)
  // NOTE: individual element accesses can be slow; avoid when possible and prefer batch operations
  A(1,2) = 456.0;
  A.print("A:");
  
  A = 5.0;         // scalars are treated as a 1x1 matrix
  A.print("A:");
  
  A.set_size(5,5); // change the size (data is not preserved)
  
  A.fill(5.0);     // set all elements to a specific value
  A.diag().fill(10.0); // set diagonal elements to a specific value
  A.print("A:");

  // determinant
  cout << "det(A): " << det(A) << endl;
  
  // Cholesky decomposition
  cout << "chol(A): " << endl << chol(A) << endl;
  
  fmat B = A + 1; // scalar elementwise addition
  
  // submatrices
  cout << "B( span(0,2), span(3,4) ):" << endl << B( span(0,2), span(3,4) ) << endl;
  
  cout << "B( 0,3, size(3,2) ):" << endl << B( 0,3, size(3,2) ) << endl;
  
  cout << "B.row(0): " << endl << B.row(0) << endl;
  
  cout << "B.col(1): " << endl << B.col(1) << endl;
  
  // transpose
  cout << "B.t(): " << endl << B.t() << endl;
  
  // maximum from each column (traverse along rows)
  cout << "max(B): " << endl << max(B) << endl;
  
  // maximum from each row (traverse along columns)
  cout << "max(B,1): " << endl << max(B,1) << endl;
  
  // maximum value in B
  cout << "max(max(B)) = " << max(max(B)) << endl;
  
  // sum of each column (traverse along rows)
  cout << "sum(B): " << endl << sum(B) << endl;
  
  // sum of each row (traverse along columns)
  cout << "sum(B,1) =" << endl << sum(B, 1) << endl;
  
  // sum of all elements
  cout << "accu(B): " << accu(B) << endl;
  
  // trace = sum along diagonal
  cout << "trace(B): " << trace(B) << endl;
  
  // generate the identity matrix
  fmat C = eye<fmat>(4,4);
  
  // random matrix with values uniformly distributed in the [0,1] interval
  fmat D = randu<fmat>(4,4);
  D.print("D:");
  
  // row vectors are treated like a matrix with one row
  frowvec r(5);
  r.fill(3.5);
  r.print("r:");
  
  // column vectors are treated like a matrix with one column
  fvec q(5);
  q.fill(-1.0);
  q.print("q:");
  
  // convert matrix to vector; data in matrices is stored column-by-column
  fvec v = vectorise(A);
  v.print("v:");
  
  // dot or inner product
  cout << "as_scalar(r*q): " << as_scalar(r*q) << endl;
  
  // outer product
  cout << "q*r: " << endl << q*r << endl;
  
  // multiply-and-accumulate operation (no temporary matrices are created)
  cout << "accu(A % B) = " << accu(A % B) << endl;
  
  // example of a compound operation
  B += 2.0 * A.t();
  B.print("B:");
  
  // imat specifies an integer matrix
  // linspace will generate [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]
  imat AA = reshape(linspace<imat>(1, 9, 9), 3, 3);

  imat BB = reshape(linspace<imat>(9, 1, 9), 3, 3);
  
  // comparison of matrices (element-wise); output of a relational operator is a umat
  umat ZZ = (AA >= BB);
  ZZ.print("ZZ:");
  
  return 0;
  }

