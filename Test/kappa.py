#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, "Test/POS+MAG/SIE+SHEAR+FIXED/pert", 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(1, 0, 1)
glafic.set_lens(1, "pert", 0.261343256161012, 1.0, 20.78, 20.78, 0.004961307, -44.49485, 0.0, 0.0)
glafic.set_point(1, 1.000, 20.78691, 20.78604)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()