#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/NFW/NFW_POS', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)
glafic.set_secondary('flag_hodensity 2', verb = 0)

glafic.startup_setnum(3, 0, 1)
glafic.set_lens(1, 'anfw', 0.261343256161012, 2.085312e+11, 2.080884e+01, 2.075341e+01, 1.500761e-01, -4.431327e+00,  8.699047e+01, 0.000000e+00)
glafic.set_lens(2, 'sers', 0.261343256161012, 8e+11, 0.19174153447615125, 97.15579684060029, 51.88103345355174, 1.4424355377364537)
glafic.set_lens(3, 'sers', 0.261343256161012, 1e+11, 0.5818587320719051, 80.90881818383842, 5.979267867493113, 0.5737795483133368)
glafic.set_point(1, 1.0, 2.081271e+01, 2.077975e+01)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(2, 0, 1, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.optimize()
glafic.findimg()
glafic.writecrit(1.0)
glafic.writelens(1.0)

glafic.quit()