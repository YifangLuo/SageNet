from stiff_SGWB import LCDM_SG as sg

model = sg(r = 1e-2,
           cr = 1,
           T_re = 2e3,
           kappa10 = 1e-2,
          )

model.cosmo_param['H0'] = 68

model.SGWB_iter()