#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Write Lens/SIE_POS', -0.78, -0.78, 0.78, 0.78, 0.012, 0.012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(3, 0, 1)
glafic.set_lens(1, "sie", 0.261343256161012, 157.9333, 0.02931999999999846, -0.021570000000000533, 0.1261772, -4.523281, 0.0, 0.0)
glafic.set_lens(2, "sers", 0.261343256161012, 79706850000.0, 0.02931999999999846, -0.021570000000000533, 0.19174, 97.1557, 51.881, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 9487091000.0, 0.02931999999999846, -0.021570000000000533, 0.58185, 80.9088, 5.97926, 0.5737)
# glafic.set_lens(4)
glafic.set_point(1, 1.000, 0.03229999999999933, -0.0013299999999993872)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
# glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()