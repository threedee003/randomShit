#include "example.h"




vector<double> square(vector<double> v) {
    for(int i = 0; i < v.size(); i++) {
        v[i] = v[i] * v[i];
    }
    return v;
}



double addOne(double x) {
    return x + 1;
}