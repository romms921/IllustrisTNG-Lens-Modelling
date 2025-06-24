#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/SIE+SHEAR/SIE_POS_SHEAR', -0.78, -0.78, 0.78, 0.78, 0.012, 0.012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(4)
glafic.set_lens(1, "sie", 0.261343256161012, 157.7517, 20.81264, 20.75977, 0.1379965, 14.824, 0.0, 0.0)
glafic.set_lens(2, "sers", 0.261343256161012, 78229340000.0, 20.81264, 20.75977, 0.19174, 97.1557, 51.881, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 10269910000.0, 20.81264, 20.75977, 0.58185, 80.9088, 5.97926, 0.5737)
glafic.set_lens(4, "pert", 0.261343256161012, 1.0, 20.88501, 20.65055, 0.03150772, 135.2195, 0.0, 0.0)
glafic.set_point(1, 1.000, 20.81896, 20.77744)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()