#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

struct Element {
  double study;
  double sleep;
  double marks;
};

vector<Element> loadDataFromFile(string fileName) {
  vector<Element> data;
  string line;

  ifstream file(fileName);
  // to ignore the headings
  getline(file, line);

  while (getline(file, line)) {

    stringstream ss(line);
    Element e;
    string val;

    getline(ss, val, ',');
    e.study = stod(val);

    getline(ss, val, ',');
    e.sleep = stod(val);

    getline(ss, val, ',');
    e.marks = stod(val);

    data.push_back(e);
  }

  return data;
}

double predict(double w1, double w2, double study, double sleep, double bias) {
  return w1 * study + w2 * sleep + bias;
}

void trainNeuron(vector<Element> &data, int epochs, double learningRate) {
  // take a random value for the weights initially
  double w1 = 10, w2 = 16, bias = 6;

  for (int epoch = 1; epoch <= epochs; epoch++) {
    for (auto &item : data) {
      double predictedResult = predict(w1, w2, item.study, item.sleep, bias);
      // calculate the loss function (or error)
      double lossFunction = item.marks - predictedResult;
      w1 = w1 + learningRate * lossFunction * item.study;
      w2 = w2 + learningRate * lossFunction * item.sleep;
      bias = bias + learningRate * lossFunction * 1;
    }

    if (epoch % 100 == 0) {

      cout << "current value of weights and bias is : " << endl;
      cout << "---- epoch number: " << epoch << " -----" << endl;
      cout << "w1: " << w1 << endl;
      cout << "w2: " << w2 << endl;
      cout << "bias: " << bias << endl;
      cout << "--------" << endl;
    }
  }
  cout << "\n==============================\n";
  cout << "Training Complete!\n";
  cout << "Learned  →  w1 = " << w1 << "  w2 = " << w2 << "  bias = " << bias
       << endl;
  cout << "Expected →  w1=5  w2=3  b=4\n";
}

int main() {

  vector<Element> data = loadDataFromFile("dataset_for_neuron.csv");
  cout << "loaded " << data.size() << " examples" << endl;

  int epochs = 500;
  double learningRate = 0.01;
  trainNeuron(data, epochs, learningRate);

  return 0;
}
