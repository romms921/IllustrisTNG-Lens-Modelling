#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Write Lens/POW_POS', -0.78, -0.78, 0.78, 0.78, 0.012, 0.012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(3, 0, 1)
glafic.set_lens(1, "pow", 0.261343256161012, 1.0, 0.038069999999997606, 0.04795000000000016, 0.1395035, -9.117043, 0.2004159, 1.977167)
glafic.set_lens(2, "sers", 0.261343256161012, 51818910000.0, 0.038069999999997606, 0.04795000000000016, 0.19174, 97.1557, 2.075204, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 12175580000.0, 0.038069999999997606, 0.04795000000000016, 0.58185, 80.9088, 0.2391704, 0.5737)
# glafic.set_lens(4)
glafic.set_point(1, 1.000, 0.032159999999997524, -0.0080600000000004)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
# glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()