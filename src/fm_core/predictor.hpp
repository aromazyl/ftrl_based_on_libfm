#pragma once
#include "fm_model.h"
#include <stdint.h>
#include <iostream>
#include <string>
#define DISALLOW_COPYAND_ASSIGN(CLASSNAME) \
  CLASSNAME(const CLASSNAME&); \
  void operator=(const CLASSNAME&);

class Predictor {
  public:
    Predictor(const fm_model& fm_model) : kModel_(fm_model) {
      sum_.setSize(fm_model.num_factor);
      sum_sqr_.setSize(fm_model.num_factor);
      row_size_ = 1024;
      row_.data = (sparse_entry<FM_FLOAT>*)malloc(sizeof(sparse_entry<FM_FLOAT>) * row_size_);
      if (!row_.data) { fprintf(stderr, "row_.data malloc failure"); exit(1); }
    }
    // ~Predictor() {  free(row_.data); }
  public:
    double Predict(const sparse_row<FM_FLOAT>& x) {
      return kModel_.predict(x, sum_, sum_sqr_);
    }
    void parse(const char* buf, int len, double* label) {
      uint32_t f_num = 0;
      int offset = 0;
      row_.size = 0;
      // fprintf(stderr, "function:%s,file:%s,line:%d, buf:%s\n", __FUNCTION__, __FILE__, __LINE__, buf);
      sscanf(buf, "%lf", label);
      while (buf[offset] != '\t' && buf[offset] != ' ') ++offset;
      while (offset <= len) {
        sscanf(buf + offset, "%u:%f", &row_.data[f_num].id, &row_.data[f_num].value);
        row_.size += 1;
        ++f_num;
        if (f_num == row_size_) {
          row_.data = (sparse_entry<float>*)realloc(row_.data, sizeof(sparse_entry<FM_FLOAT>) * row_size_ * 2);
          row_size_ *= 2;
        }
        while (buf[offset] != '\t' && buf[offset] != ' '
            && buf[offset] != '\n' && buf[offset] != '\0') ++offset;
        offset ++;
      }
    }
    void PredictFromStream() {
      std::string line;
      line.reserve(2048);
      double label;
      while (std::getline(std::cin, line)) {
        parse(line.data(), line.size(), &label);
        printf("%lf\t%lf\n", label, Predict(row_));
      }
    }
  private:
    const fm_model kModel_;
    DVector<double> sum_;
    DVector<double> sum_sqr_;
    sparse_row<FM_FLOAT> row_;
    uint32_t row_size_;

  private:
    DISALLOW_COPYAND_ASSIGN(Predictor)
};
