#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

struct Student {
  string name;
  double dsa;
  double projects;
  double iq;
  double attendance;
  int placed;
};

struct normalized_stats {
  double dsa_min, dsa_max;
  double projects_min, projects_max;
  double iq_min, iq_max;
  double attendance_min, attendance_max;
};

struct model_weights {
  double dsa_w = 0.0;
  double projects_w = 0.0;
  double iq_w = 0.0;
  double attendance_w = 0.0;
  double bias = 0.0;
};

vector<Student> load_CSV(string filename) {
  vector<Student> data;
  ifstream file(filename);
  if (!file.is_open()) {
    cerr << "Could not open CSV file" << endl;
    return data;
  }
  string line;
  // skip headings from csv file
  getline(file, line);

  while (getline(file, line)) {
    stringstream ss(line);
    string token;
    Student s;

    getline(ss, s.name, ',');

    getline(ss, token, ',');
    s.dsa = stod(token);

    getline(ss, token, ',');
    s.projects = stod(token);

    getline(ss, token, ',');
    s.iq = stod(token);

    getline(ss, token, ',');
    s.attendance = stod(token);

    getline(ss, token, ',');
    s.placed = stod(token);

    data.push_back(s);
  }
  cout << "Loaded " << data.size() << " students data from " << filename
       << endl;
  return data;
}

normalized_stats compute_normalized_stats(const vector<Student> &students) {

  normalized_stats stats;
  // initialize with the first student's data
  stats.dsa_max = stats.dsa_min = students[0].dsa;
  stats.projects_max = stats.projects_min = students[0].projects;
  stats.iq_max = stats.iq_min = students[0].iq;
  stats.attendance_max = stats.attendance_min = students[0].attendance;

  for (const Student &s : students) {

    stats.dsa_max = max(s.dsa, stats.dsa_max);
    stats.dsa_min = min(s.dsa, stats.dsa_min);

    stats.projects_max = max(s.projects, stats.projects_max);
    stats.projects_min = min(s.projects, stats.projects_min);

    stats.iq_max = max(s.iq, stats.iq_max);
    stats.iq_min = min(s.iq, stats.iq_min);

    stats.attendance_max = max(s.attendance, stats.attendance_max);
    stats.attendance_min = min(s.attendance, stats.attendance_min);
  }

  cout << "\n─── Normalization Stats (from training data) ───\n";
  cout << "DSA:        [" << stats.dsa_min << ", " << stats.dsa_max << "]\n";
  cout << "Projects:   [" << stats.projects_min << ", " << stats.projects_max
       << "]\n";
  cout << "IQ:         [" << stats.iq_min << ", " << stats.iq_max << "]\n";
  cout << "Attendance: [" << stats.attendance_min << ", "
       << stats.attendance_max << "]\n";

  return stats;
}

double normalize_single_value(double value, double max, double min) {
  if (max == min) return 0;
  return (value - min) / (max - min);
}

double sigmoid(double z) {
  return (1.0 / (1.0 + exp(-z)));
}

double predict(const normalized_stats &n, const model_weights &m,
               const Student &s) {

  double n_dsa = normalize_single_value(s.dsa, n.dsa_max, n.dsa_min);

  double n_projects =
      normalize_single_value(s.projects, n.projects_max, n.projects_min);

  double n_iq = normalize_single_value(s.iq, n.iq_max, n.iq_min);

  double n_attendance =
      normalize_single_value(s.attendance, n.attendance_max, n.attendance_min);

  double prediction = m.dsa_w * n_dsa + m.projects_w * n_projects +
                      m.iq_w * n_iq + m.attendance_w * n_attendance + m.bias;

  return sigmoid(prediction);
}

double binary_cross_entropy_loss(double prediction, double actual_value) {
  double eps = 1e-15;
  // If floating-point inaccuracies ever cause P to evaluate to slightly
  // above 1.0 (e.g. 1.0000000001), 1 - Prediction + eps can become negative,
  // triggering a NaN return from log.
  prediction = max(eps, min(1.0 - eps, prediction));
  return -(actual_value * log(prediction) +
           (1 - actual_value) * log(1 - prediction));
}

void train(model_weights &m, const vector<Student> &data,
           const normalized_stats &n, double lr, int epochs) {

  cout << "\n------Training------\n";
  for (int epoch = 1; epoch <= epochs; epoch++) {
    double total_loss_in_this_epoch = 0.0;
    for (const Student &s : data) {
      double prediction = predict(n, m, s);
      total_loss_in_this_epoch +=
          binary_cross_entropy_loss(prediction, s.placed);

      double error = prediction - s.placed;

      // update the weights in this epoch
      m.dsa_w -=
          lr * error * normalize_single_value(s.dsa, n.dsa_max, n.dsa_min);
      m.projects_w -=
          lr * error *
          normalize_single_value(s.projects, n.projects_max, n.projects_min);
      m.iq_w -= lr * error * normalize_single_value(s.iq, n.iq_max, n.iq_min);
      m.attendance_w -= lr * error *
                        normalize_single_value(s.attendance, n.attendance_max,
                                               n.attendance_min);
      m.bias -= lr * error;
    }

    if (epoch % 100 == 0) {
      cout << "Epoch: " << setw(4) << epoch << "  |  Average loss: " << fixed
           << setprecision(4) << total_loss_in_this_epoch / data.size() << endl;
    }
  }
}

void evaluate(const model_weights &m, const vector<Student> &data,
              const normalized_stats &n) {
  int correct = 0;
  for (const Student &s : data) {
    double p = predict(n, m, s);
    int predicted = (p >= 0.5) ? 1 : 0;
    if (predicted == s.placed) correct++;
  }
  double accuracy_percentage = (double)correct / data.size() * 100;
  cout << "\n─── Accuracy on training data ───\n";
  cout << "Correct: " << correct << " / " << data.size() << "  (" << fixed
       << setprecision(2) << accuracy_percentage << "%)\n";
}

int main() {

  vector<Student> data = load_CSV("students.csv");
  if (data.empty()) return 1;

  // compute normalization stats FROM training data
  // no hardcoding — model figures out min/max itself
  normalized_stats norm_stat = compute_normalized_stats(data);

  // initialize model
  model_weights model;

  // train
  train(model, data, norm_stat, 0.05, 1000);

  // print final weights
  cout << "\n─── Final Weights ───\n";
  cout << "w_dsa:        " << model.dsa_w << "\n";
  cout << "w_projects:   " << model.projects_w << "\n";
  cout << "w_iq:         " << model.iq_w << "\n";
  cout << "w_attendance: " << model.attendance_w << "\n";
  cout << "bias:         " << model.bias << "\n";

  // evaluate accuracy
  evaluate(model, data, norm_stat);
  return 0;
}
