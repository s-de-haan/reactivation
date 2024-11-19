import plot
import classify
import importlib
import preprocess
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# reload when updating code
importlib.reload(preprocess)
# mouse and date

mice = ['NN8', 'NN9', 'NN11', 'NN13', 'NN16', 'NN17', 'NN23', 'NN28']
dates_per_mouse = {'NN8': ['210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'],
# dates_per_mouse = {'NN8': ['210318', '210320', '210321', '210327', '210329', '210330'],
                    'NN9': ['210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'],
                    'NN11': ['210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'],
                    'NN13': ['210811', '210812', '210813', '210814', '210815', '210816', '210817', '210818'],
                    'NN16': ['211014', '211015', '211016', '211017', '211018', '211019', '211020', '211021', '211022'],
                    'NN17': ['211025', '211026', '211028', '211029', '211030', '211031', '211101'],
                    'NN23': ['220416', '220417', '220418', '220419', '220420', '220421', '220422', '220423', '220424'],
                    'NN28': ['230210', '230211', '230212', '230214', '230216', '230217']}

for mouse in mice:
    for date in dates_per_mouse[mouse]:
        print(mouse, date)
        # create folders to save files
        paths = preprocess.create_folders(mouse, date)
        print('folders created')
        # import data for mouse and date as dict
        session_data = preprocess.load_data(paths)
        print('data loaded')
        # process and plot behavior
        behavior = preprocess.process_behavior(session_data, paths)
        print('behavior done')
        # save masks so can run in matlab to process other planes
        # preprocess.cell_masks(paths, 0)
        print('masks saved')
        # grab activity 
        deconvolved = preprocess.process_activity(paths, 'spks', 3, 0)
        print('activity processed')
        # normalize activity
        norm_deconvolved = preprocess.normalize_deconvolved(deconvolved, behavior, paths, 0)
        print('activity normalized')
        # gassuain filter acitivity
        norm_moving_deconvolved_filtered = preprocess.difference_gaussian_filter(norm_deconvolved, 4, behavior, paths, 0)
        print('activity filtered')
        # make trial averaged traces and basline subtract
        mean_cs_1_responses_df = preprocess.normalized_trial_averaged(norm_deconvolved, behavior, 'cs_1')
        mean_cs_2_responses_df = preprocess.normalized_trial_averaged(norm_deconvolved, behavior, 'cs_2')
        print('traces done')
        # get sig cells
        [cs_1_poscells, cs_1_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_1')
        [cs_2_poscells, cs_2_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_2')
        [both_poscells, both_sigcells] = preprocess.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
        print('sig cells done')
        # get idx of top cell differences
        idx = preprocess.get_index(behavior, mean_cs_1_responses_df, mean_cs_2_responses_df, cs_1_poscells, cs_2_poscells, both_poscells, both_sigcells, paths, 1)
        print('idx done')
        # get prior for synchronous cue activity
        prior = classify.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
        print('prior done')
        # logistic regression
        y_pred = classify.log_regression(behavior, norm_deconvolved, norm_moving_deconvolved_filtered, both_poscells, prior)
        print('logistic regression done')
        # process classified output
        y_pred = classify.process_classified(y_pred, prior, paths, 1)
        print('classified done')
        plot.reactivation_cue_vector(norm_deconvolved, idx, y_pred, behavior, paths, [], [])
        # plot heatmap of top cells
        # plot.sorted_map(behavior, mean_cs_1_responses_df, mean_cs_2_responses_df, idx['cs_1'].squeeze(), idx['cs_2'].squeeze(), 150, paths)
        # plot mean reactivation probability after cues
        plot.reactivation_rate(y_pred, behavior, paths, [])
        print('reactivation rate done')
        # plot reactivation bias over time
        plot.reactivation_bias(y_pred, behavior, paths, [], [])
        print('reactivation bias done')
        # plot physical evoked reactivations
        # plot.reactivation_physical(y_pred, behavior, paths, [], [])
        # plot activity change with reactivation rates over time
        plot.activity_across_trials(norm_deconvolved, behavior, y_pred, idx, paths, [], [])
        print('activity across trials done')
        # plot activity control
        plot.activity_control(norm_deconvolved, behavior, paths, [], [])
        print('activity control done')
        # plot reactivation raster
        # plot.reactivation_raster(behavior, norm_deconvolved, y_pred, idx['cs_1_df'], idx['cs_2_df'], idx['both'], paths, [])
        print('done')
        