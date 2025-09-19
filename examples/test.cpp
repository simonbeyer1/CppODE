#include "fadiff.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>

using namespace std;
using namespace fadbad;
namespace ublas = boost::numeric::ublas;

int main()
{
  constexpr int n = 3; // Dimension R^3

  // Eingabevariablen x
  ublas::vector<F<double>> x(n);
  x(0) = 1.0;
  x(1) = 2.0;
  x(2) = 3.0;

  // Markiere jede Komponente als unabhängige Variable
  for(int i=0;i<n;++i){
    x(i).diff(i,n);  // Komponente i, insgesamt n Variablen
  }

  // Funktionsausgabe f(x)
  ublas::vector<F<double>> f(n);
  f(0) = x(0)*x(0) + x(1)*x(2);    // f1
  f(1) = x(0) + x(1)*x(1);    // f2
  f(2) = x(0)*x(1)*x(2);           // f3

  // Werte der Funktion
  cout << "f(x) = [";
  for(int i=0;i<n;++i){
    cout << f(i).x() << (i<n-1? ", ":"");
  }
  cout << "]\n";

  // Jacobimatrix df/dx
  ublas::matrix<double> J(n,n);
  for(int i=0;i<n;++i){
    for(int j=0;j<n;++j){
      J(i,j) = f(i).d(j);  // df_i / dx_j
    }
  }

  cout << "Jacobian:\n";
  for(int i=0;i<n;++i){
    for(int j=0;j<n;++j){
      cout << J(i,j) << " ";
    }
    cout << "\n";
  }

  ublas::identity_matrix<F<double>> Idx(n);
  cout << "Identity:\n";
  for(int i=0;i<n;++i){
    for(int j=0;j<n;++j){
      cout << const_cast<fadbad::F<double>&>(Idx(i,j)).x() << " ";
    }
    cout << "\n";
  }

  return 0;
}
