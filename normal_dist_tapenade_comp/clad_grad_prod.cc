void gauss_grad_0_1_2(double *x, double *p, double sigma, int dim, clad::array_ref<double> _d_x, clad::array_ref<double> _d_p, clad::array_ref<double> _d_sigma) {
  int _d_dim = 0;
  double _d_t = 0;
  unsigned long _t0;
  int _d_i = 0;
  clad::tape<double> _t1 = {};
  clad::tape<int> _t2 = {};
  clad::tape<int> _t3 = {};
  clad::tape<double> _t4 = {};
  clad::tape<int> _t5 = {};
  clad::tape<int> _t6 = {};
  double _t7;
  double _t8;
  double _t9;
  double _t10;
  double _t11;
  double _t12;
  double _t13;
  double _t14;
  double _t15;
  double _t16;
  double _t17;
  double _t18;
  double _t19;
  double t = 0;
  _t0 = 0;
  for (int i = 0; i < dim; i++) {
    _t0++;
    t += clad::push(_t4, (x[clad::push(_t2, i)] - p[clad::push(_t3, i)])) * clad::push(_t1, (x[clad::push(_t5, i)] - p[clad::push(_t6, i)]));
  }
  _t8 = -t;
  _t10 = sigma;
  _t11 = 2 * _t10;
  _t9 = sigma;
  _t7 = (_t11 * _t9);
  t = _t8 / _t7;
  _t14 = 2 * 3.1415926535897931;
  _t15 = -dim / 2.;
  _t16 = std::pow(_t14, _t15);
  _t17 = sigma;
  _t13 = std::pow(_t17, -0.5);
  _t18 = _t16 * _t13;
  _t19 = t;
  _t12 = std::exp(_t19);
  double gauss_return = _t18 * _t12;
  goto _label0;
 _label0:
  {
    double _r8 = 1 * _t12;
    double _r9 = _r8 * _t13;
    double _grad0 = 0.;
    double _grad1 = 0.;
    custom_derivatives::pow_grad(_t14, _t15, &_grad0, &_grad1);
    double _r10 = _r9 * _grad0;
    double _r11 = _r10 * 3.1415926535897931;
    double _r12 = _r9 * _grad1;
    double _r13 = _r12 / 2.;
    _d_dim += -_r13;
    double _r14 = _t16 * _r8;
    double _grad2 = 0.;
    double _grad3 = 0.;
    custom_derivatives::pow_grad(_t17, -0.5, &_grad2, &_grad3);
    double _r15 = _r14 * _grad2;
    * _d_sigma += _r15;
    double _r16 = _r14 * _grad3;
    double _r17 = _t18 * 1;
    double _r18 = _r17 * custom_derivatives::exp_darg0(_t19);
    _d_t += _r18;
  }
  {
    double _r_d1 = _d_t;
    double _r2 = _r_d1 / _t7;
    _d_t += -_r2;
    double _r3 = _r_d1 * -_t8 / (_t7 * _t7);
    double _r4 = _r3 * _t9;
    double _r5 = _r4 * _t10;
    double _r6 = 2 * _r4;
    * _d_sigma += _r6;
    double _r7 = _t11 * _r3;
    * _d_sigma += _r7;
    _d_t -= _r_d1;
  }
  for (; _t0; _t0--) {
    double _r_d0 = _d_t;
    _d_t += _r_d0;
    double _r0 = _r_d0 * clad::pop(_t1);
    _d_x[clad::pop(_t2)] += _r0;
    _d_p[clad::pop(_t3)] += -_r0;
    double _r1 = clad::pop(_t4) * _r_d0;
    _d_x[clad::pop(_t5)] += _r1;
    _d_p[clad::pop(_t6)] += -_r1;
    _d_t -= _r_d0;
  }
}
