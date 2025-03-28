#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'SIE/SIE_POS+FLUX', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(1, 0, 1)
glafic.set_lens(1, 'sie', 0.2613, 1.579763e+02, 2.080977e+01, 2.075960e+01, 1.271712e-01, -4.438612e+00, 0.000000e+00, 0.000000e+00)
glafic.set_point(1, 1.000, 1.0000, 2.081284e+01)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.kapparad(1.0000, 2.081284e+01, 2.078037e+01, 0.03, 0.8, 1000, 1)
glafic.kappacum(1.0000, 2.081284e+01, 2.078037e+01, 0.03, 0.8, 1000, 1)

glafic.quit()