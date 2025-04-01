#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Kappa/NFW+SHEAR/NFW_POS+FLUX_SHEAR', 20.0, 20.0, 21.56, 21.56, 0.001, 0.001, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(2, 0, 1)
glafic.set_lens(1, 'anfw', 0.2613, 2.268691e+11, 2.081333e+01, 2.076800e+01, 6.882355e-02, 3.531638e+01, 7.998061e+01, 0.000000e+00)
glafic.set_lens(2, 'pert', 0.2613, 1.000000e+00, 2.074037e+01, 2.084955e+01, 3.262113e-02, 1.503822e+02, 0.000000e+00, 1.000000e-04)
glafic.set_point(1, 1.000, 2.081152e+01, 2.078058e+01)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.writelens(1.0)

glafic.quit()