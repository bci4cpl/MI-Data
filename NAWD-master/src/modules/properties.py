IEEE_properties = {
    # Data
    'data_dir' : './data/ieee_dataset/',
    'tmin' : 0,
    'tmax' : 6,
    'select_label' : [1, 4],
    'filterLim' : [1,40], # In Hz
    'fs' : 100,
    'amplitude_th' : 250,
    'min_trials' : 10,

    'sub_list' : ['A2', 'A3'],# 'A4', 'A5', 'A6', 'A7', 'A8','S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8','S9','S10','S11', 'S12'],

    # Model Adjustments
    'encoder_pad' : [1, 1, 1],
    'decoder_pad' : [1, 0, 1, 0, 1, 2],
    'latent_sz' : 4704

}

Shu_properties = {
    # Data
    'filterLim' : [1,40], # In Hz
    'fs' : 250,
    'sub_list' : ['{:03d}'.format(n) for n in range(1,3)], # [001 - 026]
    'data_dir' : './data/shu_dataset/',

    # Model Adjustments
    'encoder_pad' : [1, 1, 1],
    'decoder_pad' : [1, 0, 1, 0, 1, 2],
    'latent_sz' : 1504

}

sub201_properties = {
    # Data
    'data_dir' : './data/Chist_Era_dataset/',
    'sub' : '201',
    'eyes_state' : 'CC',
    'block' : [1],
    'trial_len' : 6,
    'filter_lim' : [1,40], # In Hz
    'elec_idxs' : range(11),

    # Model Adjustments
    'encoder_pad' : [1, 1, 1],
    'decoder_pad' : [1, 1, 1, 0, 1, 0],
    'latent_sz' : 1120
}

hyper_params = {
    'device' : 'cpu', # 'cuda'
    'ae_lrn_rt' : 3e-4,
    'n_epochs' : 1,
    'btch_sz' : 8,
    'cnvl_filters' : [8, 16, 32],
}

result_params = {
    'result_dir' : 'C:/Users/ofera/studies/NAWD/results',
    # 'result_dir' : './results',

    # Plotting
    'colors' :['#32CD32', '#4169E1', '#FF2400', '#702963', '#FFC300'],#['#008080', '#FFA500', '#800000', '#C83762', '#2A6D31'],#['#008080', '#FFA500', '#006400', '#800080', '#800000'],
    'alpha' : 0.25
}