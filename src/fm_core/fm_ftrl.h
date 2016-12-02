#pragma once

#include "fm_model.h"

class Model {
  public:
    Model(void* model) { this->model_ = model; }
    void* get() const { return model_; };
  private:
    void* model_;
};
class FTRL {
  public:
    FTRL(fm_model* model);
    virtual ~FTRL();
  public:
    void Update(const sparse_row<float>& sample, double label);
    void Dump(const std::string& file);
    void Load(const std::string& file);
    double predict(sparse_row<FM_FLOAT>& x) {
      return 1.0 / (1 + exp(-this->model_->predict(x)));
    }
    double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr) {
      return 1.0 / (1 + exp(-this->model_->predict(x, sum, sum_sqr)));
    }
  public:
    fm_model* model_;
    DVectorDouble vz;
    DVectorDouble vn;
    DMatrixDouble mz;
    DMatrixDouble mn;
    double kAlpha;
    double kBeta;
    double kThrinkage;
    double kL1;
    double kL2;
  private:
    double CalculateW(double z, double n);
};



FTRL::FTRL(fm_model* model) {
  this->model_ = model;
  vz.setSize(model->num_attribute);
  vn.setSize(model->num_attribute);
  mz.setSize(model->num_factor, model->num_attribute);
  mn.setSize(model->num_factor, model->num_attribute);
  vz.init(0);
  vn.init(0);
  mz.init(0, 0);
  mn.init(0, 0);
  kAlpha = 0.1;
  kBeta = 0.1;
  kThrinkage = 0.1;
  kL1 = 0.1;
  kL2 = 0.1;
}

FTRL::~FTRL() {}

void FTRL::Dump(const std::string& file) {
  vz.save(file + "/vz");
  vn.save(file + "/vn");
  mz.save(file + "/mz");
  mn.save(file + "/mn");
}


void FTRL::Load(const std::string& file) {
  vz.load(file + "/vz");
  vn.load(file + "/vn");
  mz.load(file + "/mz");
  mn.load(file + "/mn");
}

namespace {
  inline int sgn(double x) { return x >= 0 ? 1 : -1; }
}
double FTRL::CalculateW(double z, double n) {
  if (fabs(z) < kL1) {
    return 0;
  } else {
    return -(z - sgn(z) * kL1)
      / ((kBeta + sqrt(n)) / kAlpha + kL2);
  }
}
void FTRL::Update(const sparse_row<float>& sample, double label) {
  if (model_->k1) {
    for (uint i = 0; i < sample.size; ++i) {
      double& w = model_->w(sample.data[i].id);
      w = CalculateW(vz(sample.data[i].id), vn(sample.data[i].id));
    }
  }
  for (int f = 0; f < model_->num_factor; ++f) {
    for (uint i = 0; i < sample.size; ++i) {
      double& v_f_i = this->model_->v(f, sample.data[i].id);
      v_f_i = CalculateW(mz(f, sample.data[i].id), mn(f, sample.data[i].id));
    }
  }
  double p = this->model_->predict(sample);
  if (model_->k1) {
    for (uint i = 0; i < sample.size; ++i) {
      double grad = sample.data[i].value * (p - label);
      double square_g = grad * grad;
      double& n = vn(sample.data[i].id);
      double sigma = 1.0 / kAlpha * (sqrt(n + square_g) - sqrt(n));
      double& z = vz(sample.data[i].id);
      z = z + grad - sigma * model_->w(sample.data[i].id);
      n += square_g;
    }
  }
  for (int f = 0; f < model_->num_factor; ++f) {
    double tmp = 0.0;
    for (uint i = 0; i < sample.size; ++i) {
      tmp += model_->v(f, sample.data[i].id) * sample.data[i].value;
    }
    for (uint i = 0; i < sample.size; ++i) {
      double grad = (p - label) * (tmp * sample.data[i].value)
        - sample.data[i].value * sample.data[i].value * model_->v(f, sample.data[i].id);
      double square_g = grad * grad;
      double& n = mn(f, sample.data[i].id);
      double sigma = 1.0 / kAlpha * (sqrt(n + square_g) - sqrt(n));
      double& z = mz(f, sample.data[i].id);
      z = z + grad - sigma * model_->w(sample.data[i].id);
      n += square_g;
    }
  }
}
