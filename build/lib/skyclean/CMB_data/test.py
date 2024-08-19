from cmb_data import CMB_Data
import healpy as hp

map1 = CMB_Data("/home/max/CMB_plot/data/planck_simulation/ffp10_newdust_total_030_full_map.fits")

# print(map1.nside)

#print(map1.hp_alm_to_mw_alm(hp.map2alm(map1.original_hp_map), map1.nside*2).shape)

# map1.show_attributes()
# map1.hp_alm_to_mw_alm(map1.original_hp_map,map1.lmax)


# Skyclean 

# Skyfilter

# Byedust

# Dustfilter

# Seebetter

# Cbetter

