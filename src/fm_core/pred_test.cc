#define public public
#define private public
#define protected public
#include <gtest/gtest.h>
#include "predictor.hpp"

class PredTest : ::testing::Test {
  public:
    void SetUp() {
      model.loadModel("model.txt");
      model.num_factor = 8;
      model.num_attribute = 3460470;
      predictor = new Predictor(model);
    }
    void TearDown() {}
  public:
    Predictor* predictor;
    fm_model model;
};

TEST_F(PredTest, parse) {
  char* buf[] = {
    "0 1:1.2 13:1.2 14:0.5 16:0.01",
    "1 188:1.1 199:232 900:1.001 0:1.1"};

}
