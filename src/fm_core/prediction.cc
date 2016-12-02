#include "predictor.hpp"
#include <string>

int main(int argc, char* argv[]) {
  fm_model fmodel;
  fmodel.num_attribute = atoi(argv[2]);
  fmodel.num_factor = atoi(argv[3]);
  fmodel.init();
  fmodel.loadModel(std::string(argv[1]));
  Predictor predictor(fmodel);
  predictor.PredictFromStream();
  return 0;
}
