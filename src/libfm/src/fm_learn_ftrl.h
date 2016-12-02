#pragma once
#include "fm_learn.h"
#include "../../fm_core/fm_sgd.h"
#include "../../fm_core/fm_ftrl.h"

class fm_learn_ftrl: public fm_learn {
  protected:
    //DVector<double> sum, sum_sqr;
  public:
    virtual ~fm_learn_ftrl() {
      if (ftrl) {
        delete ftrl;
        ftrl = NULL;
      }
    }
    int num_iter;
    // double learn_rate;
    // DVector<double> learn_rates;		
    FTRL* ftrl;

    virtual void init() {		
      fm_learn::init();	
      // learn_rates.setSize(3);
      ftrl = new FTRL(fm);
      this->set_l1(fm->regw);
      this->set_l2(fm->regv);
    }
    void test_valid_ftrl() {
      assert(ftrl);
    }
    void set_alpha(double alpha) {
      test_valid_ftrl();
      ftrl->kAlpha = alpha;
    }
    void set_beta(double beta) {
      test_valid_ftrl();
      ftrl->kBeta = beta;
    }
    void set_thrinkage(double thrinkage) {
      test_valid_ftrl();
      ftrl->kThrinkage = thrinkage;
    }
    void set_l1(double l1) {
      test_valid_ftrl();
      ftrl->kL1 = l1;
    }
    void set_l2(double l2) {
      test_valid_ftrl();
      ftrl->kL2 = l2;
    }

    virtual void learn(Data& train, Data& test) { 
      fm_learn::learn(train, test);

      if (train.relation.dim > 0) {
        throw "relations are not supported with SGD";
      }
      std::cout.flush();
    }

    void FTRL_RUN(sparse_row<DATA_FLOAT> &x, double label) {
      if (!ftrl) ftrl = new FTRL(fm);
      this->ftrl->Update(x, label);
    }

    void debug() {
      std::cout << "num_iter=" << num_iter << std::endl;
      fm_learn::debug();
    }

    virtual void predict(Data& data, DVector<double>& out) {
      assert(data.data->getNumRows() == out.dim);
      for (data.data->begin(); !data.data->end(); data.data->next()) {
        double p = predict_case(data);
        if (task == TASK_REGRESSION ) {
          p = std::min(max_target, p);
          p = std::max(min_target, p);
        } else if (task == TASK_CLASSIFICATION) {
          p = 1.0/(1.0 + exp(-p));
        } else {
          throw "task not supported";
        }
        out(data.data->getRowIndex()) = p;
      }
    }

};
