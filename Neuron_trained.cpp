#include <iostream>
using namespace std;

double predict(double study, double sleep) {
  double w1 = 5.00134, w2 = 2.99743, bias = 3.98783;
  return (w1 * study) + (w2 * sleep) + bias;
}

int main() {

  cout << predict(10, 5) << endl;

  return 0;
}
