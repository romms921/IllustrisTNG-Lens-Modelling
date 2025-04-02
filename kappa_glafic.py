#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'kappa/EIN/EIN_POS', 20.0, 20.0, 21.56, 21.56, 0.001, 0.001, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(1, 0, 1)
glafic.set_lens(1, 'ein', 0.2613, 4.795440e+10, 2.080736e+01, 2.074175e+01, 2.012030e-01, -4.422571e+00, 4.999760e+01, 7.000000e-01)
glafic.set_point(1, 1.000, 2.081290e+01, 2.077943e+01)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.writelens(1.0)

glafic.quit()