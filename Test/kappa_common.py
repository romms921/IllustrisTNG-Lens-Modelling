#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, "Test/POS+MAG/SIE+SHEAR/SIE_POS_SHEAR_macro", 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(2, 0, 1)
glafic.set_lens(1, "sie", 0.261343256161012, 146.4924, 20.81547, 20.77724, 0.07993487, 80.27386, 0.0, 0.0)
glafic.set_lens(2, "pert", 0.261343256161012, 1.0, 20.78, 20.78, 0.02485792, 170.5445, 0.0, 0.1356432)
glafic.set_point(1, 1.000, 20.80986, 20.77816)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.writelens(1.0)

glafic.quit()