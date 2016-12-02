#ifndef FM_LEARN_FTRL_ELEMENT_H_
#define FM_LEARN_FTRL_ELEMENT_H_

#include "fm_learn_ftrl.h"
#include "../../fm_core/fm_ftrl.h"

class fm_learn_ftrl_element: public fm_learn_ftrl {
  public:
    virtual void init() {
      fm_learn_ftrl::init();

      if (log != NULL) {
        log->addField("rmse_train", std::numeric_limits<double>::quiet_NaN());
      }
    }
    virtual void learn(Data& train, Data& test) {
      fm_learn_ftrl::learn(train, test);

      // FTRL
      for (int i = 0; i < num_iter; i++) {
        double iteration_time = getusertime();
        for (train.data->begin(); !train.data->end(); train.data->next()) {
          double label = task ?
            train.target(train.data->getRowIndex()) :
            (1.0 / (1 + exp(-train.target(train.data->getRowIndex()))));
          FTRL_RUN(train.data->getRow(), label);
        }
        iteration_time = (getusertime() - iteration_time);
        double rmse_train = evaluate(train);
        double rmse_test = evaluate(test);
        std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
        if (log != NULL) {
          log->log("rmse_train", rmse_train);
          log->log("time_learn", iteration_time);
          log->newLine();
        }
      }
    }
};

#endif /*FM_LEARN_SGD_ELEMENT_H_*/
