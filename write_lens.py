#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/POW/POW_POS', -0.78, -0.78, 0.78, 0.78, 0.012, 0.012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(3, 0, 1)
glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 20.8512217, 20.8340052, 1.191710e-01, -4.440222e+00, 4.911248e-01, 1.977167)
glafic.set_lens(2, 'sers', 0.261343256161012, 8e+10, 20.8512217, 20.8340052, 0.19174, 97.1557, 51.8810, 1.4424)
glafic.set_lens(3, 'sers', 0.261343256161012, 1e+10, 20.8512217, 20.8340052, 0.58185, 80.9088, 5.97926, 0.5737)
# glafic.set_lens(4)
glafic.set_point(1, 1.0, 2.081260e+01, 2.077943e+01)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()